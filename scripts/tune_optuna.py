"""
SentryAML - M4c：用 Optuna 调 LightGBM 参数
===========================================
M4b 手工试了 8 组配置，最好的是"学习率减半"（验证集 AUC-PR 0.5998）。
这里让 Optuna 自动在更大的参数空间里搜索。

三个机制：
  1. TPE 采样（贝叶斯优化）：
     把已完成的试验按成绩分成"好的"和"差的"两组，分别估计参数分布，
     下一次优先在"好的分布概率高、差的分布概率低"的地方取值。
     比随机搜索更快集中到有希望的区域。
  2. 剪枝（Median Pruner）：
     训练过程中定期把验证集 AUC-PR 报告给 Optuna；
     如果低于之前所有试验在同一阶段的中位数，就提前终止这次试验。
     （早停看"自己有没有停止进步"，剪枝看"比别人差多少"，两者同时用。）
  3. 热启动：第一次试验直接用 M4b 的最优配置，保证起点不差。

试验记录存在 reports/optuna.db（SQLite），中途 Ctrl+C 后重新运行会接着跑。

【注意】这里在验证集上做几十次"挑最好的"，验证集成绩的乐观偏差比 M4b 更大。
         最终以测试集为准，且只选一个模型进测试集。

用法：
    python scripts/tune_optuna.py                    # 默认 40 次试验
    python scripts/tune_optuna.py --n-trials 60
    python scripts/tune_optuna.py --timeout 1800     # 最多跑 30 分钟
    python scripts/tune_optuna.py --fresh            # 删除旧记录，从头开始
"""

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import optuna

from eval_utils import compare, evaluate, print_report
from feature_spec import model_features
from train_lgbm import load_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
STUDY_NAME = "lgbm_auc_pr"

# M4b 的最优配置（学习率减半），作为第一次试验
WARM_START = {
    "learning_rate": 0.025,
    "num_leaves": 31,
    "min_data_in_leaf": 500,
    "lambda_l2": 10.0,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
}


class PruningCallback:
    """
    LightGBM 回调：每隔 interval 轮，把验证集 AUC-PR 报告给 Optuna。
    Optuna 判断该剪枝时，抛出 TrialPruned 异常，LightGBM 训练随即中止。

    自己写这个回调而不用 optuna-integration 包，一是少一个依赖，
    二是能看清剪枝到底是怎么接到训练循环里的。
    """

    def __init__(self, trial: optuna.Trial, valid_name: str = "val",
                 metric: str = "average_precision", interval: int = 25):
        self.trial = trial
        self.valid_name = valid_name
        self.metric = metric
        self.interval = interval

    def __call__(self, env: lgb.callback.CallbackEnv) -> None:
        step = env.iteration + 1
        if step % self.interval != 0:
            return
        # evaluation_result_list 的每一项：(数据集名, 指标名, 数值, 是否越大越好)
        for data_name, metric_name, value, _ in env.evaluation_result_list:
            if data_name == self.valid_name and metric_name == self.metric:
                self.trial.report(value, step)
                if self.trial.should_prune():
                    raise optuna.TrialPruned(f"第 {step} 轮 AUC-PR {value:.4f}，低于中位数")
                return


def make_objective(X_tr, y_tr, X_va, y_va, cat, seed, max_rounds, patience, best_holder):
    # 数据集只构建一次，所有试验共用，省去每次重新分箱的时间。
    # 但 LightGBM 默认会按"第一次用到的 min_data_in_leaf"预先剔除一些特征（feature_pre_filter），
    # 之后的试验如果换了更小的 min_data_in_leaf，就会报错。
    # 共用数据集、又要调 min_data_in_leaf，就必须关掉这个预过滤。
    ds_params = {"feature_pre_filter": False}
    dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat,
                         free_raw_data=False, params=ds_params)
    dval = lgb.Dataset(X_va, label=y_va, categorical_feature=cat, reference=dtrain)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary",
            "metric": "average_precision",
            "verbosity": -1,
            "seed": seed,
            "num_threads": 0,
            "bagging_freq": 1,
            # ---- 搜索空间 ----
            # 跨几个数量级的参数用 log=True：在对数尺度上均匀取样
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves":       trial.suggest_int("num_leaves", 15, 127, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 100, 2000, log=True),
            "lambda_l2":        trial.suggest_float("lambda_l2", 1e-3, 50.0, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        }

        evals = {}
        t0 = time.perf_counter()
        booster = lgb.train(
            params, dtrain,
            num_boost_round=max_rounds,
            valid_sets=[dtrain, dval],
            valid_names=["train", "val"],
            callbacks=[
                lgb.early_stopping(patience, verbose=False),
                lgb.record_evaluation(evals),
                PruningCallback(trial),
            ],
        )
        best_it = booster.best_iteration
        ap_tr = evals["train"]["average_precision"][best_it - 1]
        ap_va = evals["val"]["average_precision"][best_it - 1]

        # 附加信息存进试验记录，事后可以查
        trial.set_user_attr("best_iteration", best_it)
        trial.set_user_attr("train_ap", ap_tr)
        trial.set_user_attr("gap", ap_tr - ap_va)
        trial.set_user_attr("seconds", time.perf_counter() - t0)

        # 模型对象不能存进 SQLite，在内存里只保留目前最好的一个
        if best_holder.get("ap", -1) < ap_va:
            best_holder.update(ap=ap_va, booster=booster, params=params,
                               trial=trial.number)
        return ap_va

    return objective


def print_trial(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    """每次试验结束后打印一行。"""
    p = trial.params
    if trial.state == optuna.trial.TrialState.PRUNED:
        step = max(trial.intermediate_values) if trial.intermediate_values else 0
        val = trial.intermediate_values.get(step, float("nan"))
        print(f"  #{trial.number:<3} 剪枝  第{step:>4}轮 AP {val:.4f}  "
              f"lr={p['learning_rate']:.3f} leaves={p['num_leaves']} "
              f"min_leaf={p['min_data_in_leaf']}")
        return
    if trial.state != optuna.trial.TrialState.COMPLETE:
        return
    u = trial.user_attrs
    mark = "  <- 当前最好" if trial.number == study.best_trial.number else ""
    print(f"  #{trial.number:<3} 完成  验证AP {trial.value:.4f}  差距 {u['gap']:.3f}  "
          f"轮数 {u['best_iteration']:>4}  {u['seconds']:>4.0f}s  "
          f"lr={p['learning_rate']:.3f} leaves={p['num_leaves']} "
          f"min_leaf={p['min_data_in_leaf']} l2={p['lambda_l2']:.2g} "
          f"ff={p['feature_fraction']:.2f} bf={p['bagging_fraction']:.2f}{mark}")


def main():
    ap = argparse.ArgumentParser(description="Optuna 调参")
    ap.add_argument("--n-trials", type=int, default=40)
    ap.add_argument("--timeout", type=int, default=None, help="最长运行秒数")
    ap.add_argument("--neg-rate", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-rounds", type=int, default=3000)
    ap.add_argument("--patience", type=int, default=100)
    ap.add_argument("--fresh", action="store_true", help="删除旧的试验记录，从头开始")
    args = ap.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    db_path = REPORT_DIR / "optuna.db"
    if args.fresh and db_path.exists():
        db_path.unlink()
    storage = f"sqlite:///{db_path.as_posix()}"

    num, cat = model_features(exclude_known_leaky=True)
    cols = num + cat
    X_tr, y_tr = load_split("train", cols, cat, args.neg_rate, args.seed)
    X_va, y_va = load_split("val", cols, cat, None, args.seed)
    print(f"训练集 {len(y_tr):,} 行（洗钱 {int(y_tr.sum()):,}），"
          f"验证集 {len(y_va):,} 行（洗钱 {int(y_va.sum()):,}）")

    optuna.logging.set_verbosity(optuna.logging.WARNING)   # 关掉 Optuna 自带的冗长日志

    # 续跑时换一个种子：否则采样器会从头重放同一串随机数，
    # 前几次随机探索会和上次完全重复，白白浪费试验
    done_before = 0
    if db_path.exists():
        try:
            done_before = len(optuna.load_study(study_name=STUDY_NAME, storage=storage).trials)
        except KeyError:
            done_before = 0

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage,
        load_if_exists=True,                               # 已有记录就接着跑
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed + done_before, n_startup_trials=8),
        # 前 5 次试验不剪枝（还没有足够的比较对象）；每次试验前 200 轮不剪枝（还没学稳）
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=200),
    )
    done = len(study.trials)
    if done == 0:
        study.enqueue_trial(WARM_START)                    # 热启动
        print("第一次试验使用 M4b 最优配置（热启动）")
    else:
        print(f"接着上次继续：已有 {done} 次试验记录")

    print(f"\n开始搜索：最多 {args.n_trials} 次试验"
          + (f"，最长 {args.timeout} 秒" if args.timeout else ""))
    best_holder = {}
    t0 = time.perf_counter()
    study.optimize(
        make_objective(X_tr, y_tr, X_va, y_va, cat, args.seed,
                       args.max_rounds, args.patience, best_holder),
        n_trials=args.n_trials,
        timeout=args.timeout,
        callbacks=[print_trial],
    )
    elapsed = time.perf_counter() - t0

    # ---------------- 汇总 ----------------
    states = [t.state for t in study.trials]
    n_done = states.count(optuna.trial.TrialState.COMPLETE)
    n_pruned = states.count(optuna.trial.TrialState.PRUNED)
    best = study.best_trial
    print(f"\n共 {len(study.trials)} 次试验：完成 {n_done}，剪枝 {n_pruned}，"
          f"本次用时 {elapsed / 60:.1f} 分钟")
    print(f"最好的试验 #{best.number}：验证集 AUC-PR {best.value:.4f}，"
          f"差距 {best.user_attrs['gap']:.3f}")
    for k, v in best.params.items():
        print(f"    {k:<18} {v:.4g}" if isinstance(v, float) else f"    {k:<18} {v}")

    # 参数重要性：估计每个参数的取值对成绩的影响有多大
    try:
        imp = optuna.importance.get_param_importances(study)
        print("\n--- 参数重要性（Optuna 估计，可对照 M4b 的手工消融结论）---")
        for k, v in imp.items():
            print(f"    {k:<18} {v:6.1%}  {'█' * int(v * 40)}")
    except Exception as e:                                  # 完成的试验太少时会失败
        imp = {}
        print(f"\n（参数重要性暂时算不出来：{e}）")

    # ---------------- 最好模型的完整报告 ----------------
    # 如果最好的试验来自之前的运行（续跑时），内存里没有它的模型，需要重训一次
    if best_holder.get("trial") != best.number:
        print(f"\n最好的试验 #{best.number} 来自之前的运行，用相同参数重训一次……")
        params = {"objective": "binary", "metric": "average_precision", "verbosity": -1,
                  "seed": args.seed, "num_threads": 0, "bagging_freq": 1, **best.params}
        dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat,
                             params={"feature_pre_filter": False})
        dval = lgb.Dataset(X_va, label=y_va, categorical_feature=cat, reference=dtrain)
        booster = lgb.train(params, dtrain, num_boost_round=args.max_rounds,
                            valid_sets=[dval], valid_names=["val"],
                            callbacks=[lgb.early_stopping(args.patience, verbose=False)])
        best_holder.update(booster=booster, params=params, trial=best.number)

    booster = best_holder["booster"]
    p_va = booster.predict(X_va, num_iteration=booster.best_iteration)
    print("\n" + "=" * 60)
    print(f"Optuna 最优配置（试验 #{best.number}）在验证集上的表现")
    print("=" * 60)
    metrics = evaluate(y_va, p_va, args.neg_rate)
    print_report(metrics, args.neg_rate)

    sweep_path = REPORT_DIR / "m4b_sweep.json"
    if sweep_path.exists():
        with open(sweep_path, encoding="utf-8") as f:
            sweep = json.load(f)
        print("\n  --- 与 M4b 手工最优对比（同一验证集）---")
        compare("M4b 手工", sweep["val_metrics"], "Optuna", metrics)
        diff = metrics["auc_pr"] - sweep["val_metrics"]["auc_pr"]
        print(f"\n  AUC-PR 差 {diff:+.4f}。在验证集上挑了 {len(study.trials)} 次，"
              f"小于约 0.005 的提升很可能只是挑中了噪声。")

    # ---------------- 保存 ----------------
    booster.save_model(str(MODEL_DIR / "lgbm_optuna.txt"),
                       num_iteration=booster.best_iteration)
    # 元数据改用 JSON：任何语言都能读，也能直接打开看
    meta = {"features_num": num, "features_cat": cat, "neg_rate": args.neg_rate,
            "params": best_holder["params"], "best_iteration": booster.best_iteration,
            "optuna_trial": best.number}
    with open(MODEL_DIR / "lgbm_optuna_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    trials = [{"number": t.number, "state": t.state.name, "value": t.value,
               "params": t.params, **t.user_attrs} for t in study.trials]
    with open(REPORT_DIR / "m4c_optuna.json", "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "best_trial": best.number,
                   "best_value": best.value, "best_params": best.params,
                   "param_importance": imp, "n_complete": n_done, "n_pruned": n_pruned,
                   "val_metrics": metrics, "trials": trials},
                  f, ensure_ascii=False, indent=2, default=str)
    print(f"\n模型已保存：{MODEL_DIR / 'lgbm_optuna.txt'}")
    print(f"试验记录：{db_path}（可续跑）   报告：{REPORT_DIR / 'm4c_optuna.json'}")


if __name__ == "__main__":
    main()