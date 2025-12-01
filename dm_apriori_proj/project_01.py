import pandas as pd
from itertools import combinations
from decimal import Decimal, getcontext
import psutil
import time
import os
from pathlib import Path


def get_memory_usage():
    """
    获取当前进程的内存使用情况（KB）
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024  # 将字节转换为KB


def generate_float_range(start, stop, step):
    """
    生成一个浮点数序列，使用Decimal提高精度
    （当前代码里没用到这个函数，保留以防后续扩展）
    """
    getcontext().prec = 10  # 设置Decimal的精度
    start = Decimal(start)
    stop = Decimal(stop)
    step = Decimal(step)
    while start < stop:
        yield float(start)
        start += step


def generate_initial_candidates(dataset):
    """
    生成初始候选项集 initial_candidates
    """
    initial_candidates = set()
    for transaction in dataset:
        for item in transaction:
            initial_candidates.add(frozenset([item]))
    # 排序只是为了结果稳定，便于对比
    return sorted(initial_candidates)


def calculate_support(dataset, candidates, min_support):
    """
    扫描数据集，计算候选项集的支持度
    """
    item_count = {}
    for transaction in dataset:
        for candidate in candidates:
            if candidate.issubset(transaction):
                item_count[candidate] = item_count.get(candidate, 0) + 1

    total_transactions = float(len(dataset))
    frequent_itemsets = []
    support_data = {}
    for itemset in item_count:
        support = item_count[itemset] / total_transactions
        if support >= min_support:
            frequent_itemsets.append(itemset)
        support_data[itemset] = support
    return frequent_itemsets, support_data


def has_infrequent_subset(candidate, previous_frequent_itemsets):
    """
    检查候选项集是否包含非频繁项集
    """
    subsets = [frozenset(x) for x in combinations(candidate, len(candidate) - 1)]
    for subset in subsets:
        if subset not in previous_frequent_itemsets:
            return True
    return False


def generate_candidate_itemsets(previous_frequent_itemsets, k):
    """
    由频繁项集生成候选项集 Ck，并进行剪枝
    """
    candidate_itemsets = []
    length = len(previous_frequent_itemsets)
    for i in range(length):
        for j in range(i + 1, length):
            # 取前 k-2 项，判断是否可以连接
            first_itemsets = list(previous_frequent_itemsets[i])[:k - 2]
            second_itemsets = list(previous_frequent_itemsets[j])[:k - 2]
            first_itemsets.sort()
            second_itemsets.sort()
            if first_itemsets == second_itemsets:  # 前 k-2 项相同则合并
                candidate = previous_frequent_itemsets[i] | previous_frequent_itemsets[j]
                if not has_infrequent_subset(candidate, previous_frequent_itemsets):
                    candidate_itemsets.append(candidate)
    return candidate_itemsets


def generate_association_rules(frequent_itemsets, support_data, min_confidence):
    """
    从频繁项集中生成关联规则
    """
    rules = []
    for itemset in frequent_itemsets:
        # itemset 至少两个元素才可能生成规则
        if len(itemset) < 2:
            continue
        for i in range(1, len(itemset)):
            for antecedent in combinations(itemset, i):
                antecedent = frozenset(antecedent)
                consequent = itemset - antecedent
                if antecedent and consequent:
                    antecedent_support = support_data.get(antecedent, 0)
                    item_support = support_data.get(itemset, 0)
                    confidence = item_support / antecedent_support if antecedent_support > 0 else 0
                    if confidence >= min_confidence:
                        rules.append((antecedent, consequent, item_support, confidence))
    return rules


def apriori_algorithm(dataset, min_support, min_confidence):
    """
    Apriori算法主函数
    """
    start_time = time.time()
    start_memory = get_memory_usage()

    candidate_c1 = generate_initial_candidates(dataset)
    transactions = list(map(set, dataset))
    frequent_l1, support_data = calculate_support(transactions, candidate_c1, min_support)
    frequent_itemsets = [frequent_l1]
    k = 2

    while len(frequent_itemsets[k - 2]) > 0:
        candidate_ck = generate_candidate_itemsets(frequent_itemsets[k - 2], k)
        frequent_lk, support_k = calculate_support(transactions, candidate_ck, min_support)
        support_data.update(support_k)
        frequent_itemsets.append(frequent_lk)
        k += 1

    # 展平所有频繁项集
    all_frequent_itemsets = [item for sublist in frequent_itemsets for item in sublist]

    # 生成关联规则
    rules = generate_association_rules(all_frequent_itemsets, support_data, min_confidence)

    end_time = time.time()
    end_memory = get_memory_usage()

    time_consumed = end_time - start_time
    memory_consumed = end_memory - start_memory

    return all_frequent_itemsets, rules, time_consumed, memory_consumed, support_data


# ======================== 路径设置（适配 macOS） ========================

# 当前脚本所在目录
BASE_DIR = Path(__file__).resolve().parent

# 数据集路径：./data/Groceries_dataset.csv
data_dir = BASE_DIR / "data"
file_path = data_dir / "Groceries_dataset.csv"

# 结果输出目录：./results/
results_dir = BASE_DIR / "results"
results_dir.mkdir(exist_ok=True)


# ======================== 数据预处理 ========================

print(f"Loading dataset from: {file_path}")
data = pd.read_csv(file_path)

# 删除 Date 列（对频繁项集挖掘无直接作用）
if "Date" in data.columns:
    data = data.drop(columns=['Date'])

# 按 Member_number 聚合，将同一会员的商品合并为一个事务
transactions = data.groupby('Member_number')['itemDescription'].apply(list).tolist()


# ======================== 参数设置 ========================

min_sup_list = [0.05, 0.1]
min_conf_list = [0.3, 0.4]

# 结果存储
support_results = []
rules_results = []
time_results = []


# ======================== 运行 Apriori 实验 ========================

for min_support in min_sup_list:
    for min_confidence in min_conf_list:
        print(f"Now Processing -> Min_Sup: {min_support}, Min_Conf: {min_confidence}")
        freq_sets, rules, time_spent, memory_spent, support_data = apriori_algorithm(
            transactions, min_support, min_confidence
        )
        print(f"Freq Itemsets Count: {len(freq_sets)}, Rules Count: {len(rules)}")

        # 存储支持度结果
        for itemset, support in support_data.items():
            support_results.append((min_support, min_confidence, set(itemset), support))

        # 存储关联规则结果
        for rule in rules:
            support = rule[2]
            confidence = rule[3]
            rules_results.append(
                (min_support, min_confidence, set(rule[0]), set(rule[1]), support, confidence)
            )

        # 存储时间与内存结果
        time_results.append(
            (min_support, min_confidence, len(freq_sets), len(rules), time_spent, memory_spent)
        )


# ======================== 保存结果到 CSV ========================

# 保存支持度结果
support_df = pd.DataFrame(
    support_results,
    columns=["Min_Support", "Min_Confidence", "Itemset", "Support"]
)
support_csv_path = results_dir / "support_results.csv"
support_df.to_csv(support_csv_path, index=False)

# 保存关联规则结果
rules_df = pd.DataFrame(
    rules_results,
    columns=["Min_Support", "Min_Confidence", "Antecedent", "Consequent", "Support", "Confidence"]
)
rules_csv_path = results_dir / "rules_results.csv"
rules_df.to_csv(rules_csv_path, index=False)

# 保存时间和内存消耗结果
time_df = pd.DataFrame(
    time_results,
    columns=[
        "Min_Support",
        "Min_Confidence",
        "Freq_Itemsets_Count",
        "Rules_Count",
        "Time(s)",
        "Memory(KB)",
    ],
)
time_csv_path = results_dir / "time_results.csv"
time_df.to_csv(time_csv_path, index=False)

print("Processing Completed. Results Saved Successfully.")
print(f"Support results saved to: {support_csv_path}")
print(f"Rules results saved to:   {rules_csv_path}")
print(f"Time/Memory results saved to: {time_csv_path}")
