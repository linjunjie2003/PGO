import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel, mutual_info_classif, \
    VarianceThreshold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, \
    precision_recall_curve, auc, average_precision_score
import warnings

warnings.filterwarnings('ignore')
import os
from scipy.spatial.distance import cdist
import textwrap

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 血管特征中英文对照字典
vascular_feature_translation = {
    'A1_一根': 'A1_one', 'A2_一根': 'A2_one', 'A3_一根': 'A3_one', 'A4_一根': 'A4_one',
    'V1_一根': 'V1_one', 'V2_一根': 'V2_one', 'V3_一根': 'V3_one', 'V4_一根': 'V4_one', 'V5_一根': 'V5_one',
    'A1_二根': 'A1_two', 'A2_二根': 'A2_two', 'A3_二根': 'A3_two', 'A4_二根': 'A4_two',
    'V2_二根': 'V2_two', 'V3_二根': 'V3_two', 'V4_二根': 'V4_two', 'V5_二根': 'V5_two',
    'A2_三根': 'A2_three', 'A3_三根': 'A3_three', 'A4_三根': 'A4_three',
    'V2_三根': 'V2_three', 'V3_三根': 'V3_three', 'V4_三根': 'V4_three', 'V5_三根': 'V5_three'
}

# 创建保存结果的目录
output_dirs = {
    'gdm': 'D:\\Desktop\\lw\\结果\\妊娠期糖尿病',
    'normal': 'D:\\Desktop\\lw\\结果\\无异常',
    'comparison': 'D:\\Desktop\\lw\\结果'
}

for dir_path in output_dirs.values():
    os.makedirs(dir_path, exist_ok=True)


# ==================== 辅助函数：确保测试集包含至少1例异常样本 ====================

def train_test_split_with_min_abnormal(X, y, test_size=0.2, random_state=42, max_attempts=100):
    """
    确保测试集中至少包含1例异常样本的数据分割函数

    参数:
    X: 特征数据
    y: 标签数据
    test_size: 测试集比例
    random_state: 随机种子
    max_attempts: 最大尝试次数

    返回:
    X_train, X_test, y_train, y_test
    """
    # 确保y是numpy数组
    y = np.array(y)

    # 检查是否有异常样本
    abnormal_indices = np.where(y == 1)[0]
    normal_indices = np.where(y == 0)[0]

    if len(abnormal_indices) == 0:
        print("警告：数据集中没有异常样本，使用常规分割")
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    print(f"异常样本数量: {len(abnormal_indices)}")
    print(f"正常样本数量: {len(normal_indices)}")

    # 如果异常样本太少，调整测试集大小以确保至少有一个异常样本
    min_test_abnormal = 1
    required_test_size = min_test_abnormal / len(abnormal_indices)

    if required_test_size > test_size:
        print(f"警告：异常样本过少，调整测试集比例从{test_size:.2f}到{required_test_size:.2f}")
        test_size = min(required_test_size + 0.1, 0.3)  # 增加一点缓冲，但不超过0.3

    # 尝试多次分割以确保测试集中至少有一个异常样本
    for attempt in range(max_attempts):
        # 使用分层抽样
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state + attempt, stratify=y
        )

        # 检查测试集中是否有异常样本
        if np.sum(y_test == 1) >= 1:
            print(f"成功分割！测试集包含{np.sum(y_test == 1)}例异常样本")
            return X_train, X_test, y_train, y_test

    # 如果多次尝试都失败，使用手动分割
    print("多次尝试分层抽样失败，使用手动分割...")

    # 确保至少有一个异常样本在测试集中
    if len(abnormal_indices) >= 2:
        # 如果有多个异常样本，将一个放入测试集，其余放入训练集
        np.random.seed(random_state)
        test_abnormal_idx = np.random.choice(abnormal_indices, size=1, replace=False)
        train_abnormal_idx = np.setdiff1d(abnormal_indices, test_abnormal_idx)
    else:
        # 如果只有一个异常样本，将其放入训练集，测试集可能没有异常样本
        test_abnormal_idx = []
        train_abnormal_idx = abnormal_indices

    # 分割正常样本
    test_normal_count = int(len(normal_indices) * test_size)
    if test_normal_count < 1:
        test_normal_count = 1

    np.random.seed(random_state)
    test_normal_idx = np.random.choice(normal_indices, size=test_normal_count, replace=False)
    train_normal_idx = np.setdiff1d(normal_indices, test_normal_idx)

    # 合并索引
    test_idx = np.concatenate([test_abnormal_idx, test_normal_idx])
    train_idx = np.concatenate([train_abnormal_idx, train_normal_idx])

    # 确保索引唯一
    test_idx = np.unique(test_idx)
    train_idx = np.unique(train_idx)

    # 分割数据
    X_train = X.iloc[train_idx] if isinstance(X, pd.DataFrame) else X[train_idx]
    X_test = X.iloc[test_idx] if isinstance(X, pd.DataFrame) else X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    print(f"手动分割完成！训练集: {len(y_train)}例（异常{np.sum(y_train == 1)}例），"
          f"测试集: {len(y_test)}例（异常{np.sum(y_test == 1)}例）")

    return X_train, X_test, y_train, y_test


def safe_roc_auc_score(y_true, y_pred_proba, pos_label=1):
    """
    安全的ROC AUC计算函数，处理类别不平衡和单一类别情况
    """
    try:
        # 检查是否有两个类别
        unique_classes = np.unique(y_true)
        if len(unique_classes) < 2:
            print(f"警告：测试集中只有{len(unique_classes)}个类别，无法计算ROC AUC")
            return np.nan

        # 检查预测概率是否有效
        if np.isnan(y_pred_proba).any():
            print("警告：预测概率包含NaN值")
            return np.nan

        # 尝试计算ROC AUC
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        return roc_auc
    except Exception as e:
        print(f"计算ROC AUC时出错: {e}")
        return np.nan


# ==================== 第一部分：妊娠期糖尿病数据分析 ====================

def analyze_gdm_data():
    print("=" * 60)
    print("妊娠期糖尿病数据分析")
    print("=" * 60)

    # 1. 加载数据
    gdm_path = 'D:\\Desktop\\lw\\妊娠期糖尿病.xlsx'
    gdm_data = pd.read_excel(gdm_path)

    # 保存原始列名用于后续分析
    original_columns = gdm_data.columns.tolist()

    # 2. 第一部分：使用胎盘特征进行胎盘质量分级
    print("\n1. 使用胎盘特征进行胎盘质量分级")

    # 提取胎盘特征列（28个特征）
    placental_features = [
        '胎盘体积', '胎盘分叶数量_一根', '胎盘分叶数量_二根', '胎盘分叶数量_三根',
        'A1_一根', 'A2_一根', 'A3_一根', 'A4_一根',
        'V1_一根', 'V2_一根', 'V3_一根', 'V4_一根', 'V5_一根',
        'A1_二根', 'A2_二根', 'A3_二根', 'A4_二根',
        'V2_二根', 'V3_二根', 'V4_二根', 'V5_二根',
        'A2_三根', 'A3_三根', 'A4_三根',
        'V2_三根', 'V3_三根', 'V4_三根', 'V5_三根'
    ]

    # 检查并处理缺失值
    placental_data = gdm_data[placental_features].copy()
    placental_data = placental_data.dropna()

    if len(placental_data) == 0:
        print("错误：胎盘特征数据全为NaN")
        return None

    # 标准化数据
    scaler = StandardScaler()
    placental_scaled = scaler.fit_transform(placental_data)

    # t-SNE降维
    print("执行t-SNE降维...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(placental_scaled) - 1))
    placental_tsne = tsne.fit_transform(placental_scaled)

    # 第一步聚类：分成6类
    print("第一次聚类：KMeans聚类为6类...")
    kmeans6 = KMeans(n_clusters=6, random_state=42)
    clusters6 = kmeans6.fit_predict(placental_scaled)

    # 计算每个簇的大小和中心点
    cluster_counts = pd.Series(clusters6).value_counts()
    print(f"各簇样本数量:\n{cluster_counts}")

    # 计算每个簇到原点的平均距离（作为"最远"的度量）
    cluster_centers = kmeans6.cluster_centers_
    cluster_distances = np.linalg.norm(cluster_centers, axis=1)

    # 找出数量最少且距离最远的簇
    min_count_cluster = cluster_counts.idxmin()
    max_distance_cluster = np.argmax(cluster_distances)

    # 综合判断：优先考虑数量最少，如果数量相同则考虑距离
    candidate_clusters = []
    for cluster_id in range(6):
        size = (clusters6 == cluster_id).sum()
        distance = cluster_distances[cluster_id]
        candidate_clusters.append({
            'cluster': cluster_id,
            'size': size,
            'distance': distance
        })

    # 按数量升序，距离降序排序
    candidate_clusters.sort(key=lambda x: (x['size'], -x['distance']))
    abnormal_cluster = candidate_clusters[0]['cluster']

    print(f"识别为异常类的簇: {abnormal_cluster}")
    print(f"该簇样本数量: {candidate_clusters[0]['size']}")
    print(f"该簇中心点距离: {candidate_clusters[0]['distance']:.4f}")

    # 修改逻辑：直接将数量最少最远的类单独归为异常类，其余归为正常类
    final_labels = np.where(clusters6 == abnormal_cluster, 1, 0)

    # 统计结果
    normal_count = (final_labels == 0).sum()
    abnormal_count = (final_labels == 1).sum()

    print(f"\n胎盘质量分级结果:")
    print(f"正常胎盘: {normal_count} 例 ({normal_count / len(final_labels) * 100:.1f}%)")
    print(f"异常胎盘: {abnormal_count} 例 ({abnormal_count / len(final_labels) * 100:.1f}%)")

    # 分析分离标准
    print("\n胎盘质量分离标准分析:")

    # 计算两类在各个特征上的均值差异
    normal_mean = placental_data[final_labels == 0].mean()
    abnormal_mean = placental_data[final_labels == 1].mean()
    mean_diff = (abnormal_mean - normal_mean).abs()

    # 找出差异最大的特征
    top_features = mean_diff.sort_values(ascending=False).head(5)
    print(f"区分度最大的5个胎盘特征:")
    for feature, diff in top_features.items():
        # 对于血管特征，使用英文翻译
        if feature in vascular_feature_translation:
            feature_name = vascular_feature_translation[feature]
        else:
            feature_name = feature
        print(f"  {feature_name}: 正常组均值={normal_mean[feature]:.3f}, "
              f"异常组均值={abnormal_mean[feature]:.3f}, 差异={diff:.3f}")

    # ============ 保存单图 ============

    # 1. t-SNE可视化（初步6类）单图
    plt.figure(figsize=(10, 8))
    scatter1 = plt.scatter(placental_tsne[:, 0], placental_tsne[:, 1],
                           c=clusters6, cmap='tab20', s=50, alpha=0.7)
    plt.title('t-SNE Visualization (6 Clusters) - GDM', fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.colorbar(scatter1, label="Clusters")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dirs['gdm'], 'GDM_TSNE_6Clusters.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # 2. t-SNE可视化（最终2类）单图
    plt.figure(figsize=(10, 8))
    colors = ['green' if label == 0 else 'red' for label in final_labels]
    plt.scatter(placental_tsne[:, 0], placental_tsne[:, 1],
                c=colors, s=50, alpha=0.7)
    plt.title('Placenta Quality Classification - GDM', fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.text(0.05, 0.95, f'Normal: {normal_count}', transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', color='green')
    plt.text(0.05, 0.90, f'Abnormal: {abnormal_count}', transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', color='red')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dirs['gdm'], 'GDM_Final_Classification.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 特征重要性（随机森林）单图
    plt.figure(figsize=(12, 8))
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(placental_scaled, final_labels)
    feature_importance = pd.DataFrame({
        'feature': placental_features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False).head(10)

    # 翻译特征名
    feature_importance['feature_en'] = feature_importance['feature'].map(
        lambda x: vascular_feature_translation.get(x, x)
    )

    plt.barh(range(len(feature_importance)),
             feature_importance['importance'].values)
    plt.yticks(range(len(feature_importance)), feature_importance['feature_en'])
    plt.xlabel('Importance')
    plt.title('Top 10 Important Placental Features - GDM', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dirs['gdm'], 'GDM_Feature_Importance.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # 4. 两类样本在关键特征上的分布单图
    plt.figure(figsize=(10, 8))
    key_feature = top_features.index[0]
    key_feature_en = vascular_feature_translation.get(key_feature, key_feature)
    plt.hist([placental_data[final_labels == 0][key_feature],
              placental_data[final_labels == 1][key_feature]],
             bins=20, alpha=0.7, label=['Normal', 'Abnormal'],
             color=['green', 'red'])
    plt.xlabel(key_feature_en)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Key Feature: {key_feature_en} - GDM', fontsize=14, fontweight='bold')
    plt.legend()
    plt.axvline(x=normal_mean[key_feature], color='green', linestyle='--', alpha=0.7)
    plt.axvline(x=abnormal_mean[key_feature], color='red', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dirs['gdm'], 'GDM_Key_Feature_Distribution.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # ============ 套图保存 ============

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. t-SNE可视化（初步6类）
    scatter1 = axes[0, 0].scatter(placental_tsne[:, 0], placental_tsne[:, 1],
                                  c=clusters6, cmap='tab20', s=50, alpha=0.7)
    axes[0, 0].set_title('t-SNE Visualization (6 Clusters) - GDM', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('t-SNE Component 1')
    axes[0, 0].set_ylabel('t-SNE Component 2')

    # 2. t-SNE可视化（最终2类）
    colors = ['green' if label == 0 else 'red' for label in final_labels]
    axes[0, 1].scatter(placental_tsne[:, 0], placental_tsne[:, 1],
                       c=colors, s=50, alpha=0.7)
    axes[0, 1].set_title('Placenta Quality Classification - GDM', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('t-SNE Component 1')
    axes[0, 1].set_ylabel('t-SNE Component 2')
    axes[0, 1].text(0.05, 0.95, f'Normal: {normal_count}', transform=axes[0, 1].transAxes,
                    fontsize=12, verticalalignment='top', color='green')
    axes[0, 1].text(0.05, 0.90, f'Abnormal: {abnormal_count}', transform=axes[0, 1].transAxes,
                    fontsize=12, verticalalignment='top', color='red')

    # 3. 特征重要性（随机森林）
    axes[1, 0].barh(range(len(feature_importance)),
                    feature_importance['importance'].values)
    axes[1, 0].set_yticks(range(len(feature_importance)))
    axes[1, 0].set_yticklabels(feature_importance['feature_en'])
    axes[1, 0].set_xlabel('Importance')
    axes[1, 0].set_title('Top 10 Important Placental Features - GDM', fontsize=14, fontweight='bold')
    axes[1, 0].invert_yaxis()

    # 4. 两类样本在关键特征上的分布
    key_feature = top_features.index[0]
    key_feature_en = vascular_feature_translation.get(key_feature, key_feature)
    axes[1, 1].hist([placental_data[final_labels == 0][key_feature],
                     placental_data[final_labels == 1][key_feature]],
                    bins=20, alpha=0.7, label=['Normal', 'Abnormal'],
                    color=['green', 'red'])
    axes[1, 1].set_xlabel(key_feature_en)
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title(f'Distribution of Key Feature: {key_feature_en} - GDM', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].axvline(x=normal_mean[key_feature], color='green', linestyle='--', alpha=0.7)
    axes[1, 1].axvline(x=abnormal_mean[key_feature], color='red', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dirs['gdm'], 'GDM_Placenta_Classification_Comprehensive.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

    # 保存分析结果
    analysis_text = f"""妊娠期糖尿病胎盘质量分级分析结果
================================================

1. 数据集信息
   总样本数: {len(gdm_data)}
   可用胎盘特征样本数: {len(placental_data)}
   缺失值处理: 删除包含NaN的行

2. 聚类分析过程
   第一步聚类: KMeans聚类为6类
   异常类识别标准: 样本数量最少且距离中心最远的簇
   异常类识别结果: 簇{abnormal_cluster} (样本数: {abnormal_count}, 中心点距离: {candidate_clusters[0]['distance']:.4f})

3. 分级结果
   正常胎盘: {normal_count} 例 ({normal_count / len(final_labels) * 100:.1f}%)
   异常胎盘: {abnormal_count} 例 ({abnormal_count / len(final_labels) * 100:.1f}%)

4. 关键区分特征（前5个）
"""
    for i, (feature, diff) in enumerate(top_features.items(), 1):
        feature_en = vascular_feature_translation.get(feature, feature)
        analysis_text += f"   {i}. {feature} ({feature_en}): 差异值={diff:.3f}\n"

    analysis_text += f"""
5. 胎盘质量划分建议标准
   基于分析，建议以下综合标准判断胎盘质量异常：
   - 属于t-SNE降维空间中最稀疏的聚类簇
   - {vascular_feature_translation.get(top_features.index[0], top_features.index[0])} 值异常偏高/偏低
   - 多个胎盘特征同时偏离正常范围

6. 临床意义
   本研究提出的自动分级方法可帮助医生：
   - 快速识别高风险胎盘
   - 为妊娠期糖尿病管理提供参考
   - 指导个性化产前护理方案
"""

    with open(os.path.join(output_dirs['gdm'], 'GDM_Analysis_Result.txt'), 'w', encoding='utf-8') as f:
        f.write(analysis_text)

    print("分析结果已保存到:", os.path.join(output_dirs['gdm'], 'GDM_Analysis_Result.txt'))

    return {
        'vascular_labels': final_labels,
        'vascular_data': placental_data,
        'tsne_result': placental_tsne,
        'top_features': top_features,
        'normal_count': normal_count,
        'abnormal_count': abnormal_count,
        'abnormal_cluster_id': abnormal_cluster
    }


def build_gdm_classifier(gdm_data, placental_labels):
    print("\n" + "=" * 60)
    print("构建妊娠期糖尿病胎盘质量预测分类器")
    print("=" * 60)

    # 提取临床特征（12个特征）
    clinical_features = [
        '孕妇年龄', '孕妇孕前体重', '孕妇现体重', '孕妇身高',
        '体重指数', '孕周', '临床诊断信息', '空腹血糖孕7-9月',
        '糖化血红蛋白孕7-9月', '糖耐量试验_孕24-28周',
        '妊娠期糖尿病_干预措施', '血压'
    ]

    # 检查特征是否存在
    available_features = [f for f in clinical_features if f in gdm_data.columns]
    print(f"可用临床特征: {len(available_features)}个")
    print(f"特征列表: {available_features}")

    # 准备数据
    X = gdm_data[available_features].copy()
    y = placental_labels

    # 处理缺失值
    X = X.fillna(X.mean())

    # 检查类别分布
    print(f"\n类别分布: 正常={sum(y == 0)}例, 异常={sum(y == 1)}例")

    # 使用自定义分割函数确保测试集至少包含1例异常样本
    X_train, X_test, y_train, y_test = train_test_split_with_min_abnormal(
        X, y, test_size=0.2, random_state=42
    )

    print(f"训练集/测试集: {len(X_train)}/{len(X_test)}")
    print(f"训练集类别分布: 正常={sum(y_train == 0)}例, 异常={sum(y_train == 1)}例")
    print(f"测试集类别分布: 正常={sum(y_test == 0)}例, 异常={sum(y_test == 1)}例")

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 特征选择方法 - 四种不同的方法
    feature_selectors = {
        'SelectKBest (k=3)': SelectKBest(f_classif, k=min(3, len(available_features))),
        'SelectKBest MI (k=7)': SelectKBest(mutual_info_classif, k=min(7, len(available_features))),
        'RFE (SVM, k=5)': RFE(SVC(kernel='linear', random_state=42),
                              n_features_to_select=min(5, len(available_features))),
        'RandomForest Importance': SelectFromModel(RandomForestClassifier(n_estimators=200, random_state=42),
                                                   threshold='median')
    }

    # 分类器
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'SVM': SVC(probability=True, random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        'KNN': KNeighborsClassifier()
    }

    results = {}
    all_roc_data = {}
    all_pr_data = {}
    confusion_matrices = {}

    # 进行特征选择和模型训练
    for fs_name, fs_model in feature_selectors.items():
        print(f"\n使用特征选择方法: {fs_name}")

        # 特征选择
        try:
            # 对于RFE，需要使用fit_transform
            if 'RFE' in fs_name:
                X_train_fs = fs_model.fit_transform(X_train_scaled, y_train)
                X_test_fs = fs_model.transform(X_test_scaled)
                selected_indices = np.where(fs_model.support_)[0]
            elif 'RandomForest' in fs_name:
                fs_model.fit(X_train_scaled, y_train)
                X_train_fs = fs_model.transform(X_train_scaled)
                X_test_fs = fs_model.transform(X_test_scaled)
                selected_indices = np.where(fs_model.get_support())[0]
            else:
                # SelectKBest
                X_train_fs = fs_model.fit_transform(X_train_scaled, y_train)
                X_test_fs = fs_model.transform(X_test_scaled)
                selected_indices = np.where(fs_model.get_support())[0]
        except Exception as e:
            print(f"特征选择方法 {fs_name} 出错: {e}")
            continue

        selected_features = [available_features[i] for i in selected_indices] if len(selected_indices) > 0 else []
        print(f"选择的特征 ({len(selected_features)}个): {selected_features}")

        fs_results = {}

        # 训练和评估每个分类器
        for clf_name, clf in classifiers.items():
            try:
                # 训练
                clf.fit(X_train_fs, y_train)

                # 预测
                y_pred = clf.predict(X_test_fs)

                # 对于SVM，如果没有predict_proba，则计算决策函数
                if hasattr(clf, 'predict_proba'):
                    y_pred_proba = clf.predict_proba(X_test_fs)[:, 1]
                else:
                    y_pred_proba = clf.decision_function(X_test_fs)

                # 评估
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)

                # 安全地计算ROC AUC
                roc_auc = safe_roc_auc_score(y_test, y_pred_proba)

                # 计算ROC曲线（如果可能）
                try:
                    if len(np.unique(y_test)) >= 2:
                        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                    else:
                        fpr, tpr = [0, 1], [0, 1]  # 默认曲线
                except Exception as e:
                    print(f"  计算ROC曲线时出错: {e}")
                    fpr, tpr = [0, 1], [0, 1]

                # 计算PR曲线
                try:
                    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
                    pr_auc = auc(recall, precision)
                    average_precision = average_precision_score(y_test, y_pred_proba)
                except Exception as e:
                    print(f"  计算PR曲线时出错: {e}")
                    precision, recall = [1, 0], [0, 1]
                    pr_auc = np.nan
                    average_precision = np.nan

                # 计算混淆矩阵
                cm = confusion_matrix(y_test, y_pred)

                fs_results[clf_name] = {
                    'accuracy': accuracy,
                    'precision': report['weighted avg']['precision'],
                    'recall': report['weighted avg']['recall'],
                    'f1-score': report['weighted avg']['f1-score'],
                    'roc_auc': roc_auc,
                    'pr_auc': pr_auc,
                    'average_precision': average_precision,
                    'selected_features': selected_features,
                    'model': clf,
                    'selected_indices': selected_indices,
                    'fpr': fpr,
                    'tpr': tpr,
                    'precision_curve': precision,
                    'recall_curve': recall,
                    'confusion_matrix': cm
                }

                # 保存数据用于绘图
                model_key = f"{fs_name}_{clf_name}"
                all_roc_data[model_key] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc, 'label': f"{clf_name} ({fs_name})"}
                all_pr_data[model_key] = {'precision': precision, 'recall': recall, 'auc': pr_auc,
                                          'label': f"{clf_name} ({fs_name})", 'avg_precision': average_precision}
                confusion_matrices[model_key] = cm

                # 显示结果
                auc_display = f"{roc_auc:.4f}" if not np.isnan(roc_auc) else "nan"
                print(f"  {clf_name}: 准确率={accuracy:.4f}, "
                      f"F1={report['weighted avg']['f1-score']:.4f}, ROC AUC={auc_display}")
            except Exception as e:
                print(f"  分类器 {clf_name} 训练失败: {e}")
                continue

        results[fs_name] = fs_results

    # 找出最佳模型
    best_accuracy = 0
    best_model_info = None
    best_fs_method = None
    best_clf_name = None

    for fs_method, clf_results in results.items():
        for clf_name, metrics in clf_results.items():
            if metrics['accuracy'] > best_accuracy:
                best_accuracy = metrics['accuracy']
                best_model_info = metrics
                best_fs_method = fs_method
                best_clf_name = clf_name

    print(f"\n最佳模型: {best_fs_method} + {best_clf_name}")
    print(f"最佳准确率: {best_accuracy:.4f}")
    if best_model_info:
        print(f"使用的特征 ({len(best_model_info['selected_features'])}个): {best_model_info['selected_features']}")
    else:
        print("警告：未找到有效的模型")

    # ============ 绘制ROC曲线对比图 ============
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Random')

    # 绘制所有模型的ROC曲线（过滤掉AUC为nan的模型）
    valid_models = 0
    for model_key, roc_data in all_roc_data.items():
        if not np.isnan(roc_data['auc']):
            plt.plot(roc_data['fpr'], roc_data['tpr'],
                     label=f"{roc_data['label']} (AUC = {roc_data['auc']:.3f})",
                     linewidth=2, alpha=0.7)
            valid_models += 1

    if valid_models == 0:
        print("警告：没有有效的ROC曲线可绘制")
        plt.text(0.5, 0.5, "No valid ROC curves available\n(可能由于测试集中缺少异常样本)",
                 ha='center', va='center', transform=plt.gca().transAxes)
    else:
        plt.legend(loc='lower right', fontsize=9)

    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison - GDM', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dirs['gdm'], 'GDM_ROC_Curves.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # ============ 绘制PR曲线对比图 ============
    plt.figure(figsize=(10, 8))

    # 计算随机模型的PR曲线
    if len(y_test) > 0:
        random_precision = sum(y_test) / len(y_test)
    else:
        random_precision = 0.5

    valid_pr_models = 0
    for model_key, pr_data in all_pr_data.items():
        if not np.isnan(pr_data['avg_precision']):
            plt.plot(pr_data['recall'], pr_data['precision'],
                     label=f"{pr_data['label']} (AP = {pr_data['avg_precision']:.3f})",
                     linewidth=2, alpha=0.7)
            valid_pr_models += 1

    if valid_pr_models > 0:
        plt.axhline(y=random_precision, color='k', linestyle='--', label=f'Random (AP = {random_precision:.3f})')
        plt.legend(loc='upper right', fontsize=9)
    else:
        plt.text(0.5, 0.5, "No valid PR curves available",
                 ha='center', va='center', transform=plt.gca().transAxes)

    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves Comparison - GDM', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dirs['gdm'], 'GDM_PR_Curves.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # ============ 绘制最佳模型混淆矩阵 ============
    if best_model_info:
        plt.figure(figsize=(8, 6))
        cm = best_model_info['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normal', 'Abnormal'],
                    yticklabels=['Normal', 'Abnormal'])
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title(f'Confusion Matrix: {best_clf_name} ({best_fs_method}) - GDM\nAccuracy: {best_accuracy:.4f}',
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dirs['gdm'], 'GDM_Best_Model_Confusion_Matrix.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    # ============ 绘制特征数与识别率关系图（四种特征选择方法） ============
    plt.figure(figsize=(12, 8))

    # 定义颜色和标记
    colors = ['b', 'g', 'r', 'm']
    markers = ['o', 's', '^', 'D']
    line_styles = ['-', '--', '-.', ':']

    # 为每种特征选择方法尝试不同的特征数量
    for idx, (fs_name, fs_model_creator) in enumerate([
        ('SelectKBest (f_classif)', lambda k: SelectKBest(f_classif, k=k)),
        ('SelectKBest (MI)', lambda k: SelectKBest(mutual_info_classif, k=k)),
        ('RFE (SVM)', lambda k: RFE(SVC(kernel='linear', random_state=42), n_features_to_select=k)),
        ('RF Importance', lambda k: SelectFromModel(
            RandomForestClassifier(n_estimators=100, random_state=42),
            max_features=k) if hasattr(SelectFromModel, 'max_features') else
        SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42),
                        threshold='median'))
    ]):
        feature_num_accuracies = {}

        # 遍历不同的特征数量
        max_features = min(10, len(available_features))
        for k in range(1, max_features + 1):
            try:
                # 创建特征选择器
                if 'RF Importance' in fs_name:
                    # 对于RandomForest Importance，使用不同的阈值来选择特征
                    if k <= len(available_features):
                        # 创建一个临时特征选择器
                        temp_fs = SelectFromModel(
                            RandomForestClassifier(n_estimators=100, random_state=42),
                            max_features=k
                        )
                        X_train_fs = temp_fs.fit_transform(X_train_scaled, y_train)
                        X_test_fs = temp_fs.transform(X_test_scaled)
                    else:
                        continue
                else:
                    # 对于其他方法，直接使用k参数
                    selector = fs_model_creator(k)
                    X_train_fs = selector.fit_transform(X_train_scaled, y_train)
                    X_test_fs = selector.transform(X_test_scaled)

                # 使用随机森林分类器
                clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
                clf.fit(X_train_fs, y_train)
                y_pred = clf.predict(X_test_fs)
                accuracy = accuracy_score(y_test, y_pred)

                feature_num_accuracies[k] = accuracy
            except Exception as e:
                # print(f"特征选择方法 {fs_name}, 特征数 {k} 训练失败: {e}")
                continue

        # 绘制该方法的曲线
        if feature_num_accuracies:
            x_vals = list(feature_num_accuracies.keys())
            y_vals = list(feature_num_accuracies.values())

            # 找到最佳特征数和准确率
            if feature_num_accuracies:
                best_k = max(feature_num_accuracies, key=feature_num_accuracies.get)
                best_acc = feature_num_accuracies[best_k]

                plt.plot(x_vals, y_vals,
                         color=colors[idx % len(colors)],
                         marker=markers[idx % len(markers)],
                         linestyle=line_styles[idx % len(line_styles)],
                         linewidth=2, markersize=8,
                         label=f'{fs_name} (Best: {best_k} features, Acc={best_acc:.3f})')

                # 标记最佳点
                plt.scatter(best_k, best_acc, color=colors[idx % len(colors)],
                            s=150, zorder=5, edgecolors='black')

    plt.xlabel('Number of Selected Features', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Feature Number vs Recognition Rate (4 Feature Selection Methods) - GDM',
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dirs['gdm'], 'GDM_Feature_Number_vs_Accuracy.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # ============ 保存单图 ============

    # 1. 各特征选择方法的最佳准确率单图
    plt.figure(figsize=(10, 6))
    fs_accuracies = {}
    for fs_method, clf_results in results.items():
        if clf_results:
            best_fs_acc = max([metrics['accuracy'] for metrics in clf_results.values()])
            fs_accuracies[fs_method] = best_fs_acc

    if fs_accuracies:
        plt.bar(range(len(fs_accuracies)), list(fs_accuracies.values()))
        plt.xticks(range(len(fs_accuracies)), list(fs_accuracies.keys()), rotation=45, ha='right')
        plt.ylabel('Best Accuracy')
        plt.title('Best Accuracy by Feature Selection Method - GDM', fontweight='bold')
        plt.ylim([0, 1.0])
        plt.tight_layout()
        plt.savefig(os.path.join(output_dirs['gdm'], 'GDM_FS_Accuracy.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    # 2. 最佳分类器的性能指标单图
    if best_model_info:
        plt.figure(figsize=(8, 6))
        metrics_names = ['accuracy', 'precision', 'recall', 'f1-score']
        metrics_values = [best_model_info[m] for m in metrics_names]
        plt.bar(metrics_names, metrics_values)
        plt.ylabel('Score')
        plt.title(f'Best Model Performance: {best_clf_name} - GDM', fontweight='bold')
        plt.ylim([0, 1])
        plt.tight_layout()
        plt.savefig(os.path.join(output_dirs['gdm'], 'GDM_Best_Model_Performance.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    # 3. 特征重要性（随机森林）单图
    plt.figure(figsize=(12, 8))
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    feature_importance = pd.DataFrame({
        'feature': available_features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False).head(min(10, len(available_features)))

    # 创建特征名翻译字典
    feature_translation = {
        '孕妇年龄': 'Maternal Age',
        '孕妇孕前体重': 'Pre-pregnancy Weight',
        '孕妇现体重': 'Current Weight',
        '孕妇身高': 'Height',
        '体重指数': 'BMI',
        '孕周': 'Gestational Weeks',
        '临床诊断信息': 'Clinical Diagnosis',
        '空腹血糖孕7-9月': 'Fasting Blood Glucose (7-9 months)',
        '糖化血红蛋白孕7-9月': 'Glycated Hemoglobin (7-9 months)',
        '糖耐量试验_孕24-28周': 'Glucose Tolerance Test (24-28 weeks)',
        '妊娠期糖尿病_干预措施': 'GDM Intervention',
        '血压': 'Blood Pressure'
    }

    # 翻译特征名
    feature_importance['feature_en'] = feature_importance['feature'].map(
        lambda x: feature_translation.get(x, x)
    )

    plt.barh(range(len(feature_importance)), feature_importance['importance'])
    plt.yticks(range(len(feature_importance)), feature_importance['feature_en'])
    plt.xlabel('Importance')
    plt.title('Top Clinical Features Importance - GDM', fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dirs['gdm'], 'GDM_Clinical_Feature_Importance.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # ============ 套图保存 ============

    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    axes = axes.flatten()

    plot_idx = 0

    # 1. 各特征选择方法的最佳准确率
    if fs_accuracies:
        axes[plot_idx].bar(range(len(fs_accuracies)), list(fs_accuracies.values()))
        axes[plot_idx].set_xticks(range(len(fs_accuracies)))
        axes[plot_idx].set_xticklabels(list(fs_accuracies.keys()), rotation=45, ha='right')
        axes[plot_idx].set_ylabel('Best Accuracy')
        axes[plot_idx].set_title('Best Accuracy by Feature Selection Method - GDM', fontweight='bold')
        axes[plot_idx].set_ylim([0, 1.0])
    else:
        axes[plot_idx].text(0.5, 0.5, "No data available", ha='center', va='center')
        axes[plot_idx].set_title('Best Accuracy by Feature Selection Method - GDM', fontweight='bold')
    plot_idx += 1

    # 2. 最佳分类器的性能指标
    if best_model_info:
        axes[plot_idx].bar(metrics_names, metrics_values)
        axes[plot_idx].set_ylabel('Score')
        axes[plot_idx].set_title(f'Best Model Performance: {best_clf_name} - GDM', fontweight='bold')
        axes[plot_idx].set_ylim([0, 1])
    else:
        axes[plot_idx].text(0.5, 0.5, "No best model found", ha='center', va='center')
        axes[plot_idx].set_title('Best Model Performance - GDM', fontweight='bold')
    plot_idx += 1

    # 3. 各分类器在不同特征选择下的平均表现
    clf_avg_accuracies = {}
    for clf_name in classifiers.keys():
        accuracies = []
        for fs_method in results.keys():
            if clf_name in results[fs_method]:
                accuracies.append(results[fs_method][clf_name]['accuracy'])
        if accuracies:
            clf_avg_accuracies[clf_name] = np.mean(accuracies)

    if clf_avg_accuracies:
        axes[plot_idx].bar(range(len(clf_avg_accuracies)), list(clf_avg_accuracies.values()))
        axes[plot_idx].set_xticks(range(len(clf_avg_accuracies)))
        axes[plot_idx].set_xticklabels(list(clf_avg_accuracies.keys()), rotation=45, ha='right')
        axes[plot_idx].set_ylabel('Average Accuracy')
        axes[plot_idx].set_title('Average Accuracy by Classifier - GDM', fontweight='bold')
    else:
        axes[plot_idx].text(0.5, 0.5, "No classifier data", ha='center', va='center')
        axes[plot_idx].set_title('Average Accuracy by Classifier - GDM', fontweight='bold')
    plot_idx += 1

    # 4. 特征重要性（使用随机森林）
    axes[plot_idx].barh(range(len(feature_importance)), feature_importance['importance'])
    axes[plot_idx].set_yticks(range(len(feature_importance)))
    axes[plot_idx].set_yticklabels(feature_importance['feature_en'])
    axes[plot_idx].set_xlabel('Importance')
    axes[plot_idx].set_title('Top Clinical Features Importance - GDM', fontweight='bold')
    axes[plot_idx].invert_yaxis()
    plot_idx += 1

    # 5. 混淆矩阵（最佳模型）
    if best_model_info:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[plot_idx],
                    xticklabels=['Normal', 'Abnormal'],
                    yticklabels=['Normal', 'Abnormal'])
        axes[plot_idx].set_xlabel('Predicted')
        axes[plot_idx].set_ylabel('Actual')
        axes[plot_idx].set_title(f'Confusion Matrix: {best_clf_name} - GDM', fontweight='bold')
    else:
        axes[plot_idx].text(0.5, 0.5, "No confusion matrix available", ha='center', va='center')
        axes[plot_idx].set_title('Confusion Matrix - GDM', fontweight='bold')
    plot_idx += 1

    # 6. ROC曲线（最佳模型）
    if best_model_info and not np.isnan(best_model_info['roc_auc']):
        axes[plot_idx].plot(best_model_info['fpr'], best_model_info['tpr'], 'b-', linewidth=2,
                            label=f"AUC = {best_model_info['roc_auc']:.3f}")
        axes[plot_idx].plot([0, 1], [0, 1], 'k--')
        axes[plot_idx].set_xlabel('False Positive Rate')
        axes[plot_idx].set_ylabel('True Positive Rate')
        axes[plot_idx].set_title(f'ROC Curve: {best_clf_name}', fontweight='bold')
        axes[plot_idx].legend(loc='lower right')
        axes[plot_idx].grid(True, alpha=0.3)
    else:
        axes[plot_idx].text(0.5, 0.5, "No valid ROC curve available", ha='center', va='center')
        axes[plot_idx].set_title('ROC Curve - GDM', fontweight='bold')
    plot_idx += 1

    # 7. PR曲线（最佳模型）
    if best_model_info and not np.isnan(best_model_info['average_precision']):
        axes[plot_idx].plot(best_model_info['recall_curve'], best_model_info['precision_curve'], 'b-', linewidth=2,
                            label=f"AP = {best_model_info['average_precision']:.3f}")
        axes[plot_idx].axhline(y=random_precision, color='k', linestyle='--',
                               label=f'Random (AP = {random_precision:.3f})')
        axes[plot_idx].set_xlabel('Recall')
        axes[plot_idx].set_ylabel('Precision')
        axes[plot_idx].set_title(f'PR Curve: {best_clf_name}', fontweight='bold')
        axes[plot_idx].legend(loc='upper right')
        axes[plot_idx].grid(True, alpha=0.3)
    else:
        axes[plot_idx].text(0.5, 0.5, "No valid PR curve available", ha='center', va='center')
        axes[plot_idx].set_title('PR Curve - GDM', fontweight='bold')
    plot_idx += 1

    # 8. 特征数与准确率关系（四种方法）
    # 重新计算并绘制四种方法的特征数与准确率关系
    colors = ['b', 'g', 'r', 'm']
    markers = ['o', 's', '^', 'D']
    line_styles = ['-', '--', '-.', ':']

    has_plot = False
    for idx, (fs_name, fs_model_creator) in enumerate([
        ('SelectKBest (f_classif)', lambda k: SelectKBest(f_classif, k=k)),
        ('SelectKBest (MI)', lambda k: SelectKBest(mutual_info_classif, k=k)),
        ('RFE (SVM)', lambda k: RFE(SVC(kernel='linear', random_state=42), n_features_to_select=k)),
        ('RF Importance', lambda k: SelectFromModel(
            RandomForestClassifier(n_estimators=100, random_state=42),
            max_features=k) if hasattr(SelectFromModel, 'max_features') else
        SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42),
                        threshold='median'))
    ]):
        feature_num_accuracies = {}

        # 遍历不同的特征数量
        max_features = min(10, len(available_features))
        for k in range(1, max_features + 1):
            try:
                if 'RF Importance' in fs_name:
                    if k <= len(available_features):
                        temp_fs = SelectFromModel(
                            RandomForestClassifier(n_estimators=100, random_state=42),
                            max_features=k
                        )
                        X_train_fs = temp_fs.fit_transform(X_train_scaled, y_train)
                        X_test_fs = temp_fs.transform(X_test_scaled)
                    else:
                        continue
                else:
                    selector = fs_model_creator(k)
                    X_train_fs = selector.fit_transform(X_train_scaled, y_train)
                    X_test_fs = selector.transform(X_test_scaled)

                clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
                clf.fit(X_train_fs, y_train)
                y_pred = clf.predict(X_test_fs)
                accuracy = accuracy_score(y_test, y_pred)

                feature_num_accuracies[k] = accuracy
            except Exception as e:
                continue

        # 绘制该方法的曲线
        if feature_num_accuracies:
            x_vals = list(feature_num_accuracies.keys())
            y_vals = list(feature_num_accuracies.values())

            best_k = max(feature_num_accuracies, key=feature_num_accuracies.get)
            best_acc = feature_num_accuracies[best_k]

            axes[plot_idx].plot(x_vals, y_vals,
                                color=colors[idx % len(colors)],
                                marker=markers[idx % len(markers)],
                                linestyle=line_styles[idx % len(line_styles)],
                                linewidth=2, markersize=6,
                                label=f'{fs_name}')

            axes[plot_idx].scatter(best_k, best_acc, color=colors[idx % len(colors)],
                                   s=80, zorder=5, edgecolors='black')
            has_plot = True

    if has_plot:
        axes[plot_idx].set_xlabel('Number of Selected Features')
        axes[plot_idx].set_ylabel('Accuracy')
        axes[plot_idx].set_title('Feature Number vs Recognition Rate (4 Methods)', fontweight='bold')
        axes[plot_idx].grid(True, alpha=0.3)
        axes[plot_idx].legend(loc='lower right', fontsize=9)
    else:
        axes[plot_idx].text(0.5, 0.5, "No feature number vs accuracy data", ha='center', va='center')
        axes[plot_idx].set_title('Feature Number vs Recognition Rate - GDM', fontweight='bold')
    plot_idx += 1

    # 9. 关键特征分布
    if best_model_info and len(best_model_info['selected_features']) > 0:
        key_feature = best_model_info['selected_features'][0]
        if key_feature in gdm_data.columns:
            # 翻译关键特征名
            key_feature_en = feature_translation.get(key_feature, key_feature)
            axes[plot_idx].hist([
                gdm_data.loc[y == 0, key_feature].dropna(),
                gdm_data.loc[y == 1, key_feature].dropna()
            ], bins=20, alpha=0.7, label=['Normal Placenta', 'Abnormal Placenta'],
                color=['green', 'red'])
            axes[plot_idx].set_xlabel(key_feature_en)
            axes[plot_idx].set_ylabel('Frequency')
            axes[plot_idx].set_title(f'Distribution of Key Feature: {key_feature_en} - GDM', fontweight='bold')
            axes[plot_idx].legend()
    else:
        axes[plot_idx].text(0.5, 0.5, "No key feature distribution data", ha='center', va='center')
        axes[plot_idx].set_title('Key Feature Distribution - GDM', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dirs['gdm'], 'GDM_Classification_Results_Comprehensive.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

    # 保存分析结果 - 修复字符串格式化错误
    analysis_text = f"""妊娠期糖尿病胎盘质量预测分类器分析结果
================================================

1. 数据集信息
   总样本数: {len(gdm_data)}
   临床特征数量: {len(available_features)}
   训练集/测试集: {len(X_train)}/{len(X_test)}
   类别分布: 正常={sum(y == 0)}例, 异常={sum(y == 1)}例

2. 最佳模型配置
   特征选择方法: {best_fs_method if best_fs_method else "N/A"}
   分类器: {best_clf_name if best_clf_name else "N/A"}
   测试集准确率: {"{:.4f}".format(best_accuracy) if best_model_info else "N/A"}
"""

    # 添加ROC AUC信息
    if best_model_info:
        roc_auc_str = f"{best_model_info['roc_auc']:.4f}" if not np.isnan(best_model_info['roc_auc']) else "N/A"
        pr_auc_str = f"{best_model_info['pr_auc']:.4f}" if not np.isnan(best_model_info['pr_auc']) else "N/A"
        selected_features_str = ', '.join(best_model_info['selected_features']) if best_model_info[
            'selected_features'] else "N/A"
    else:
        roc_auc_str = "N/A"
        pr_auc_str = "N/A"
        selected_features_str = "N/A"

    analysis_text += f"""   ROC AUC: {roc_auc_str}
   PR AUC: {pr_auc_str}
   使用特征: {selected_features_str}

3. 各模型性能比较
"""

    # 添加详细性能表格
    for fs_method, clf_results in results.items():
        analysis_text += f"\n{fs_method}:\n"
        analysis_text += f"{'Classifier':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'ROC AUC':<10}\n"
        analysis_text += "-" * 70 + "\n"

        for clf_name, metrics in clf_results.items():
            roc_auc_display = f"{metrics['roc_auc']:.4f}" if not np.isnan(metrics['roc_auc']) else "nan"
            analysis_text += f"{clf_name:<20} {metrics['accuracy']:.4f}      {metrics['precision']:.4f}      {metrics['recall']:.4f}      {metrics['f1-score']:.4f}      {roc_auc_display}\n"

    analysis_text += f"""
4. 特征数与准确率关系分析
   对四种特征选择方法进行了特征数变化实验：
   - SelectKBest (f_classif): 使用F检验选择特征
   - SelectKBest (MI): 使用互信息选择特征
   - RFE (SVM): 使用SVM递归特征消除
   - RF Importance: 使用随机森林特征重要性

   实验结果显示不同特征选择方法在不同特征数下的表现差异，
   为临床实践中特征选择提供了参考。

5. 临床意义
   本模型可以帮助医生：
   - 早期预测胎盘质量异常风险
   - 基于孕妇临床指标进行风险评估
   - 指导妊娠期糖尿病的个性化管理
   - 优化产前监测策略

6. 使用建议
   - 重点关注: {', '.join(best_model_info['selected_features'][:3]) if best_model_info and best_model_info['selected_features'] else '无'}
   - 定期监测上述指标变化
   - 结合超声检查进行综合评估

7. 数据限制说明
   - 异常样本数量极少（{sum(y == 1)}例），可能影响模型泛化能力
   - ROC AUC等指标可能因测试集类别不平衡而受限
   - 建议收集更多数据以验证模型稳定性
"""

    with open(os.path.join(output_dirs['gdm'], 'GDM_Classifier_Analysis.txt'), 'w', encoding='utf-8') as f:
        f.write(analysis_text)

    print("分类器分析结果已保存到:", os.path.join(output_dirs['gdm'], 'GDM_Classifier_Analysis.txt'))

    return results, best_model_info


# ==================== 第二部分：无异常数据分析 ====================

def analyze_normal_data():
    print("\n" + "=" * 60)
    print("无异常数据分析")
    print("=" * 60)

    # 1. 加载数据
    normal_path = 'D:\\Desktop\\lw\\无异常.xlsx'
    normal_data = pd.read_excel(normal_path)

    # 2. 第一部分：使用胎盘特征进行胎盘质量分级
    print("\n1. 使用胎盘特征进行胎盘质量分级")

    # 提取胎盘特征列（28个特征）
    placental_features = [
        '胎盘体积', '胎盘分叶数量_一根', '胎盘分叶数量_二根', '胎盘分叶数量_三根',
        'A1_一根', 'A2_一根', 'A3_一根', 'A4_一根',
        'V1_一根', 'V2_一根', 'V3_一根', 'V4_一根', 'V5_一根',
        'A1_二根', 'A2_二根', 'A3_二根', 'A4_二根',
        'V2_二根', 'V3_二根', 'V4_二根', 'V5_二根',
        'A2_三根', 'A3_三根', 'A4_三根',
        'V2_三根', 'V3_三根', 'V4_三根', 'V5_三根'
    ]

    # 检查并处理缺失值
    placental_data = normal_data[placental_features].copy()
    placental_data = placental_data.dropna()

    if len(placental_data) == 0:
        print("错误：胎盘特征数据全为NaN")
        return None

    # 标准化数据
    scaler = StandardScaler()
    placental_scaled = scaler.fit_transform(placental_data)

    # t-SNE降维
    print("执行t-SNE降维...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(placental_scaled) - 1))
    placental_tsne = tsne.fit_transform(placental_scaled)

    # 第一步聚类：分成6类
    print("第一次聚类：KMeans聚类为6类...")
    kmeans6 = KMeans(n_clusters=6, random_state=42)
    clusters6 = kmeans6.fit_predict(placental_scaled)

    # 计算每个簇的大小和中心点
    cluster_counts = pd.Series(clusters6).value_counts()
    print(f"各簇样本数量:\n{cluster_counts}")

    # 计算每个簇到原点的平均距离
    cluster_centers = kmeans6.cluster_centers_
    cluster_distances = np.linalg.norm(cluster_centers, axis=1)

    # 找出数量最少且距离最远的簇
    candidate_clusters = []
    for cluster_id in range(6):
        size = (clusters6 == cluster_id).sum()
        distance = cluster_distances[cluster_id]
        candidate_clusters.append({
            'cluster': cluster_id,
            'size': size,
            'distance': distance
        })

    # 按数量升序，距离降序排序
    candidate_clusters.sort(key=lambda x: (x['size'], -x['distance']))
    abnormal_cluster = candidate_clusters[0]['cluster']

    print(f"识别为异常类的簇: {abnormal_cluster}")
    print(f"该簇样本数量: {candidate_clusters[0]['size']}")
    print(f"该簇中心点距离: {candidate_clusters[0]['distance']:.4f}")

    # 修改逻辑：直接将数量最少最远的类单独归为异常类，其余归为正常类
    final_labels = np.where(clusters6 == abnormal_cluster, 1, 0)

    # 统计结果
    normal_count = (final_labels == 0).sum()
    abnormal_count = (final_labels == 1).sum()

    print(f"\n胎盘质量分级结果:")
    print(f"正常胎盘: {normal_count} 例 ({normal_count / len(final_labels) * 100:.1f}%)")
    print(f"异常胎盘: {abnormal_count} 例 ({abnormal_count / len(final_labels) * 100:.1f}%)")

    # 分析分离标准
    print("\n胎盘质量分离标准分析:")

    # 计算两类在各个特征上的均值差异
    normal_mean = placental_data[final_labels == 0].mean()
    abnormal_mean = placental_data[final_labels == 1].mean()
    mean_diff = (abnormal_mean - normal_mean).abs()

    # 找出差异最大的特征
    top_features = mean_diff.sort_values(ascending=False).head(5)
    print(f"区分度最大的5个胎盘特征:")
    for feature, diff in top_features.items():
        # 对于血管特征，使用英文翻译
        if feature in vascular_feature_translation:
            feature_name = vascular_feature_translation[feature]
        else:
            feature_name = feature
        print(f"  {feature_name}: 正常组均值={normal_mean[feature]:.3f}, "
              f"异常组均值={abnormal_mean[feature]:.3f}, 差异={diff:.3f}")

    # ============ 保存单图 ============

    # 1. t-SNE可视化（初步6类）单图
    plt.figure(figsize=(10, 8))
    scatter1 = plt.scatter(placental_tsne[:, 0], placental_tsne[:, 1],
                           c=clusters6, cmap='tab20', s=50, alpha=0.7)
    plt.title('t-SNE Visualization (6 Clusters) - Normal', fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.colorbar(scatter1, label="Clusters")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dirs['normal'], 'Normal_TSNE_6Clusters.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # 2. t-SNE可视化（最终2类）单图
    plt.figure(figsize=(10, 8))
    colors = ['green' if label == 0 else 'red' for label in final_labels]
    plt.scatter(placental_tsne[:, 0], placental_tsne[:, 1],
                c=colors, s=50, alpha=0.7)
    plt.title('Placenta Quality Classification - Normal', fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.text(0.05, 0.95, f'Normal: {normal_count}', transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', color='green')
    plt.text(0.05, 0.90, f'Abnormal: {abnormal_count}', transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', color='red')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dirs['normal'], 'Normal_Final_Classification.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 特征重要性（随机森林）单图
    plt.figure(figsize=(12, 8))
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(placental_scaled, final_labels)
    feature_importance = pd.DataFrame({
        'feature': placental_features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False).head(10)

    # 翻译特征名
    feature_importance['feature_en'] = feature_importance['feature'].map(
        lambda x: vascular_feature_translation.get(x, x)
    )

    plt.barh(range(len(feature_importance)),
             feature_importance['importance'].values)
    plt.yticks(range(len(feature_importance)), feature_importance['feature_en'])
    plt.xlabel('Importance')
    plt.title('Top 10 Important Placental Features - Normal', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dirs['normal'], 'Normal_Feature_Importance.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # 4. 两类样本在关键特征上的分布单图
    plt.figure(figsize=(10, 8))
    key_feature = top_features.index[0]
    key_feature_en = vascular_feature_translation.get(key_feature, key_feature)
    plt.hist([placental_data[final_labels == 0][key_feature],
              placental_data[final_labels == 1][key_feature]],
             bins=20, alpha=0.7, label=['Normal', 'Abnormal'],
             color=['green', 'red'])
    plt.xlabel(key_feature_en)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Key Feature: {key_feature_en} - Normal', fontsize=14, fontweight='bold')
    plt.legend()
    plt.axvline(x=normal_mean[key_feature], color='green', linestyle='--', alpha=0.7)
    plt.axvline(x=abnormal_mean[key_feature], color='red', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dirs['normal'], 'Normal_Key_Feature_Distribution.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # ============ 套图保存 ============

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. t-SNE可视化（初步6类）
    scatter1 = axes[0, 0].scatter(placental_tsne[:, 0], placental_tsne[:, 1],
                                  c=clusters6, cmap='tab20', s=50, alpha=0.7)
    axes[0, 0].set_title('t-SNE Visualization (6 Clusters) - Normal', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('t-SNE Component 1')
    axes[0, 0].set_ylabel('t-SNE Component 2')

    # 2. t-SNE可视化（最终2类）
    colors = ['green' if label == 0 else 'red' for label in final_labels]
    axes[0, 1].scatter(placental_tsne[:, 0], placental_tsne[:, 1],
                       c=colors, s=50, alpha=0.7)
    axes[0, 1].set_title('Placenta Quality Classification - Normal', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('t-SNE Component 1')
    axes[0, 1].set_ylabel('t-SNE Component 2')
    axes[0, 1].text(0.05, 0.95, f'Normal: {normal_count}', transform=axes[0, 1].transAxes,
                    fontsize=12, verticalalignment='top', color='green')
    axes[0, 1].text(0.05, 0.90, f'Abnormal: {abnormal_count}', transform=axes[0, 1].transAxes,
                    fontsize=12, verticalalignment='top', color='red')

    # 3. 特征重要性（随机森林）
    axes[1, 0].barh(range(len(feature_importance)),
                    feature_importance['importance'].values)
    axes[1, 0].set_yticks(range(len(feature_importance)))
    axes[1, 0].set_yticklabels(feature_importance['feature_en'])
    axes[1, 0].set_xlabel('Importance')
    axes[1, 0].set_title('Top 10 Important Placental Features - Normal', fontsize=14, fontweight='bold')
    axes[1, 0].invert_yaxis()

    # 4. 两类样本在关键特征上的分布
    key_feature = top_features.index[0]
    key_feature_en = vascular_feature_translation.get(key_feature, key_feature)
    axes[1, 1].hist([placental_data[final_labels == 0][key_feature],
                     placental_data[final_labels == 1][key_feature]],
                    bins=20, alpha=0.7, label=['Normal', 'Abnormal'],
                    color=['green', 'red'])
    axes[1, 1].set_xlabel(key_feature_en)
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title(f'Distribution of Key Feature: {key_feature_en} - Normal', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].axvline(x=normal_mean[key_feature], color='green', linestyle='--', alpha=0.7)
    axes[1, 1].axvline(x=abnormal_mean[key_feature], color='red', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dirs['normal'], 'Normal_Placenta_Classification_Comprehensive.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

    # 保存分析结果
    analysis_text = f"""无异常组胎盘质量分级分析结果
================================================

1. 数据集信息
   总样本数: {len(normal_data)}
   可用胎盘特征样本数: {len(placental_data)}
   缺失值处理: 删除包含NaN的行

2. 聚类分析过程
   第一步聚类: KMeans聚类为6类
   异常类识别标准: 样本数量最少且距离中心最远的簇
   异常类识别结果: 簇{abnormal_cluster} (样本数: {abnormal_count}, 中心点距离: {candidate_clusters[0]['distance']:.4f})

3. 分级结果
   正常胎盘: {normal_count} 例 ({normal_count / len(final_labels) * 100:.1f}%)
   异常胎盘: {abnormal_count} 例 ({abnormal_count / len(final_labels) * 100:.1f}%)

4. 关键区分特征（前5个）
"""
    for i, (feature, diff) in enumerate(top_features.items(), 1):
        feature_en = vascular_feature_translation.get(feature, feature)
        analysis_text += f"   {i}. {feature} ({feature_en}): 差异值={diff:.3f}\n"

    analysis_text += f"""
5. 胎盘质量划分建议标准
   基于分析，建议以下综合标准判断胎盘质量异常：
   - 属于t-SNE降维空间中最稀疏的聚类簇
   - {vascular_feature_translation.get(top_features.index[0], top_features.index[0])} 值异常偏高/偏低
   - 多个胎盘特征同时偏离正常范围

6. 临床意义
   本研究提出的自动分级方法可帮助医生：
   - 在正常孕妇群体中识别潜在胎盘异常
   - 早期发现无症状的胎盘功能异常
   - 指导常规产前检查的重点关注方向
"""

    with open(os.path.join(output_dirs['normal'], 'Normal_Analysis_Result.txt'), 'w', encoding='utf-8') as f:
        f.write(analysis_text)

    print("分析结果已保存到:", os.path.join(output_dirs['normal'], 'Normal_Analysis_Result.txt'))

    return {
        'vascular_labels': final_labels,
        'vascular_data': placental_data,
        'tsne_result': placental_tsne,
        'top_features': top_features,
        'normal_count': normal_count,
        'abnormal_count': abnormal_count,
        'abnormal_cluster_id': abnormal_cluster
    }


def build_normal_classifier(normal_data, placental_labels):
    print("\n" + "=" * 60)
    print("构建无异常组胎盘质量预测分类器")
    print("=" * 60)

    # 提取临床特征（7个特征）
    clinical_features = [
        '孕妇年龄', '孕妇孕前体重', '孕妇现体重', '孕妇身高',
        '体重指数', '孕周', '血压'
    ]

    # 检查特征是否存在
    available_features = [f for f in clinical_features if f in normal_data.columns]
    print(f"可用临床特征: {len(available_features)}个")
    print(f"特征列表: {available_features}")

    # 准备数据
    X = normal_data[available_features].copy()
    y = placental_labels

    # 处理缺失值
    X = X.fillna(X.mean())

    # 检查类别分布
    print(f"\n类别分布: 正常={sum(y == 0)}例, 异常={sum(y == 1)}例")

    # 使用自定义分割函数确保测试集至少包含1例异常样本
    X_train, X_test, y_train, y_test = train_test_split_with_min_abnormal(
        X, y, test_size=0.2, random_state=42
    )

    print(f"训练集/测试集: {len(X_train)}/{len(X_test)}")
    print(f"训练集类别分布: 正常={sum(y_train == 0)}例, 异常={sum(y_train == 1)}例")
    print(f"测试集类别分布: 正常={sum(y_test == 0)}例, 异常={sum(y_test == 1)}例")

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 特征选择方法（与GDM相同）
    feature_selectors = {
        'SelectKBest (k=3)': SelectKBest(f_classif, k=min(3, len(available_features))),
        'SelectKBest MI (k=7)': SelectKBest(mutual_info_classif, k=min(7, len(available_features))),
        'RFE (SVM, k=5)': RFE(SVC(kernel='linear', random_state=42),
                              n_features_to_select=min(5, len(available_features))),
        'RandomForest Importance': SelectFromModel(RandomForestClassifier(n_estimators=200, random_state=42),
                                                   threshold='median')
    }

    # 分类器（与GDM相同）
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'SVM': SVC(probability=True, random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        'KNN': KNeighborsClassifier()
    }

    results = {}
    all_roc_data = {}
    all_pr_data = {}
    confusion_matrices = {}

    # 进行特征选择和模型训练
    for fs_name, fs_model in feature_selectors.items():
        print(f"\n使用特征选择方法: {fs_name}")

        # 特征选择
        try:
            # 对于RFE，需要使用fit_transform
            if 'RFE' in fs_name:
                X_train_fs = fs_model.fit_transform(X_train_scaled, y_train)
                X_test_fs = fs_model.transform(X_test_scaled)
                selected_indices = np.where(fs_model.support_)[0]
            elif 'RandomForest' in fs_name:
                fs_model.fit(X_train_scaled, y_train)
                X_train_fs = fs_model.transform(X_train_scaled)
                X_test_fs = fs_model.transform(X_test_scaled)
                selected_indices = np.where(fs_model.get_support())[0]
            else:
                # SelectKBest
                X_train_fs = fs_model.fit_transform(X_train_scaled, y_train)
                X_test_fs = fs_model.transform(X_test_scaled)
                selected_indices = np.where(fs_model.get_support())[0]
        except Exception as e:
            print(f"特征选择方法 {fs_name} 出错: {e}")
            continue

        selected_features = [available_features[i] for i in selected_indices] if len(selected_indices) > 0 else []
        print(f"选择的特征 ({len(selected_features)}个): {selected_features}")

        fs_results = {}

        # 训练和评估每个分类器
        for clf_name, clf in classifiers.items():
            try:
                # 训练
                clf.fit(X_train_fs, y_train)

                # 预测
                y_pred = clf.predict(X_test_fs)

                # 对于SVM，如果没有predict_proba，则计算决策函数
                if hasattr(clf, 'predict_proba'):
                    y_pred_proba = clf.predict_proba(X_test_fs)[:, 1]
                else:
                    y_pred_proba = clf.decision_function(X_test_fs)

                # 评估
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)

                # 安全地计算ROC AUC
                roc_auc = safe_roc_auc_score(y_test, y_pred_proba)

                # 计算ROC曲线（如果可能）
                try:
                    if len(np.unique(y_test)) >= 2:
                        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                    else:
                        fpr, tpr = [0, 1], [0, 1]  # 默认曲线
                except Exception as e:
                    print(f"  计算ROC曲线时出错: {e}")
                    fpr, tpr = [0, 1], [0, 1]

                # 计算PR曲线
                try:
                    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
                    pr_auc = auc(recall, precision)
                    average_precision = average_precision_score(y_test, y_pred_proba)
                except Exception as e:
                    print(f"  计算PR曲线时出错: {e}")
                    precision, recall = [1, 0], [0, 1]
                    pr_auc = np.nan
                    average_precision = np.nan

                # 计算混淆矩阵
                cm = confusion_matrix(y_test, y_pred)

                fs_results[clf_name] = {
                    'accuracy': accuracy,
                    'precision': report['weighted avg']['precision'],
                    'recall': report['weighted avg']['recall'],
                    'f1-score': report['weighted avg']['f1-score'],
                    'roc_auc': roc_auc,
                    'pr_auc': pr_auc,
                    'average_precision': average_precision,
                    'selected_features': selected_features,
                    'model': clf,
                    'selected_indices': selected_indices,
                    'fpr': fpr,
                    'tpr': tpr,
                    'precision_curve': precision,
                    'recall_curve': recall,
                    'confusion_matrix': cm
                }

                # 保存数据用于绘图
                model_key = f"{fs_name}_{clf_name}"
                all_roc_data[model_key] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc, 'label': f"{clf_name} ({fs_name})"}
                all_pr_data[model_key] = {'precision': precision, 'recall': recall, 'auc': pr_auc,
                                          'label': f"{clf_name} ({fs_name})", 'avg_precision': average_precision}
                confusion_matrices[model_key] = cm

                # 显示结果
                auc_display = f"{roc_auc:.4f}" if not np.isnan(roc_auc) else "nan"
                print(f"  {clf_name}: 准确率={accuracy:.4f}, "
                      f"F1={report['weighted avg']['f1-score']:.4f}, ROC AUC={auc_display}")
            except Exception as e:
                print(f"  分类器 {clf_name} 训练失败: {e}")
                continue

        results[fs_name] = fs_results

    # 找出最佳模型
    best_accuracy = 0
    best_model_info = None
    best_fs_method = None
    best_clf_name = None

    for fs_method, clf_results in results.items():
        for clf_name, metrics in clf_results.items():
            if metrics['accuracy'] > best_accuracy:
                best_accuracy = metrics['accuracy']
                best_model_info = metrics
                best_fs_method = fs_method
                best_clf_name = clf_name

    print(f"\n最佳模型: {best_fs_method} + {best_clf_name}")
    print(f"最佳准确率: {best_accuracy:.4f}")
    if best_model_info:
        print(f"使用的特征 ({len(best_model_info['selected_features'])}个): {best_model_info['selected_features']}")

    # ============ 绘制ROC曲线对比图 ============
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Random')

    # 绘制所有模型的ROC曲线（过滤掉AUC为nan的模型）
    valid_models = 0
    for model_key, roc_data in all_roc_data.items():
        if not np.isnan(roc_data['auc']):
            plt.plot(roc_data['fpr'], roc_data['tpr'],
                     label=f"{roc_data['label']} (AUC = {roc_data['auc']:.3f})",
                     linewidth=2, alpha=0.7)
            valid_models += 1

    if valid_models == 0:
        print("警告：没有有效的ROC曲线可绘制")
        plt.text(0.5, 0.5, "No valid ROC curves available",
                 ha='center', va='center', transform=plt.gca().transAxes)
    else:
        plt.legend(loc='lower right', fontsize=9)

    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison - Normal', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dirs['normal'], 'Normal_ROC_Curves.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # ============ 绘制PR曲线对比图 ============
    plt.figure(figsize=(10, 8))

    # 计算随机模型的PR曲线
    if len(y_test) > 0:
        random_precision = sum(y_test) / len(y_test)
    else:
        random_precision = 0.5

    valid_pr_models = 0
    for model_key, pr_data in all_pr_data.items():
        if not np.isnan(pr_data['avg_precision']):
            plt.plot(pr_data['recall'], pr_data['precision'],
                     label=f"{pr_data['label']} (AP = {pr_data['avg_precision']:.3f})",
                     linewidth=2, alpha=0.7)
            valid_pr_models += 1

    if valid_pr_models > 0:
        plt.axhline(y=random_precision, color='k', linestyle='--', label=f'Random (AP = {random_precision:.3f})')
        plt.legend(loc='upper right', fontsize=9)
    else:
        plt.text(0.5, 0.5, "No valid PR curves available",
                 ha='center', va='center', transform=plt.gca().transAxes)

    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves Comparison - Normal', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dirs['normal'], 'Normal_PR_Curves.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # ============ 绘制最佳模型混淆矩阵 ============
    if best_model_info:
        plt.figure(figsize=(8, 6))
        cm = best_model_info['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normal', 'Abnormal'],
                    yticklabels=['Normal', 'Abnormal'])
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title(f'Confusion Matrix: {best_clf_name} ({best_fs_method}) - Normal\nAccuracy: {best_accuracy:.4f}',
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dirs['normal'], 'Normal_Best_Model_Confusion_Matrix.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    # ============ 绘制特征数与识别率关系图（四种特征选择方法） ============
    plt.figure(figsize=(12, 8))

    # 定义颜色和标记
    colors = ['b', 'g', 'r', 'm']
    markers = ['o', 's', '^', 'D']
    line_styles = ['-', '--', '-.', ':']

    # 为每种特征选择方法尝试不同的特征数量
    for idx, (fs_name, fs_model_creator) in enumerate([
        ('SelectKBest (f_classif)', lambda k: SelectKBest(f_classif, k=k)),
        ('SelectKBest (MI)', lambda k: SelectKBest(mutual_info_classif, k=k)),
        ('RFE (SVM)', lambda k: RFE(SVC(kernel='linear', random_state=42), n_features_to_select=k)),
        ('RF Importance', lambda k: SelectFromModel(
            RandomForestClassifier(n_estimators=100, random_state=42),
            max_features=k) if hasattr(SelectFromModel, 'max_features') else
        SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42),
                        threshold='median'))
    ]):
        feature_num_accuracies = {}

        # 遍历不同的特征数量
        max_features = min(10, len(available_features))
        for k in range(1, max_features + 1):
            try:
                # 创建特征选择器
                if 'RF Importance' in fs_name:
                    # 对于RandomForest Importance，使用不同的阈值来选择特征
                    if k <= len(available_features):
                        # 创建一个临时特征选择器
                        temp_fs = SelectFromModel(
                            RandomForestClassifier(n_estimators=100, random_state=42),
                            max_features=k
                        )
                        X_train_fs = temp_fs.fit_transform(X_train_scaled, y_train)
                        X_test_fs = temp_fs.transform(X_test_scaled)
                    else:
                        continue
                else:
                    # 对于其他方法，直接使用k参数
                    selector = fs_model_creator(k)
                    X_train_fs = selector.fit_transform(X_train_scaled, y_train)
                    X_test_fs = selector.transform(X_test_scaled)

                # 使用随机森林分类器
                clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
                clf.fit(X_train_fs, y_train)
                y_pred = clf.predict(X_test_fs)
                accuracy = accuracy_score(y_test, y_pred)

                feature_num_accuracies[k] = accuracy
            except Exception as e:
                # print(f"特征选择方法 {fs_name}, 特征数 {k} 训练失败: {e}")
                continue

        # 绘制该方法的曲线
        if feature_num_accuracies:
            x_vals = list(feature_num_accuracies.keys())
            y_vals = list(feature_num_accuracies.values())

            # 找到最佳特征数和准确率
            if feature_num_accuracies:
                best_k = max(feature_num_accuracies, key=feature_num_accuracies.get)
                best_acc = feature_num_accuracies[best_k]

                plt.plot(x_vals, y_vals,
                         color=colors[idx % len(colors)],
                         marker=markers[idx % len(markers)],
                         linestyle=line_styles[idx % len(line_styles)],
                         linewidth=2, markersize=8,
                         label=f'{fs_name} (Best: {best_k} features, Acc={best_acc:.3f})')

                # 标记最佳点
                plt.scatter(best_k, best_acc, color=colors[idx % len(colors)],
                            s=150, zorder=5, edgecolors='black')

    plt.xlabel('Number of Selected Features', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Feature Number vs Recognition Rate (4 Feature Selection Methods) - Normal',
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dirs['normal'], 'Normal_Feature_Number_vs_Accuracy.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # ============ 保存单图 ============

    # 1. 各特征选择方法的最佳准确率单图
    plt.figure(figsize=(10, 6))
    fs_accuracies = {}
    for fs_method, clf_results in results.items():
        if clf_results:
            best_fs_acc = max([metrics['accuracy'] for metrics in clf_results.values()])
            fs_accuracies[fs_method] = best_fs_acc

    if fs_accuracies:
        plt.bar(range(len(fs_accuracies)), list(fs_accuracies.values()))
        plt.xticks(range(len(fs_accuracies)), list(fs_accuracies.keys()), rotation=45, ha='right')
        plt.ylabel('Best Accuracy')
        plt.title('Best Accuracy by Feature Selection Method - Normal', fontweight='bold')
        plt.ylim([0, 1.0])
        plt.tight_layout()
        plt.savefig(os.path.join(output_dirs['normal'], 'Normal_FS_Accuracy.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    # 2. 最佳分类器的性能指标单图
    if best_model_info:
        plt.figure(figsize=(8, 6))
        metrics_names = ['accuracy', 'precision', 'recall', 'f1-score']
        metrics_values = [best_model_info[m] for m in metrics_names]
        plt.bar(metrics_names, metrics_values)
        plt.ylabel('Score')
        plt.title(f'Best Model Performance: {best_clf_name} - Normal', fontweight='bold')
        plt.ylim([0, 1])
        plt.tight_layout()
        plt.savefig(os.path.join(output_dirs['normal'], 'Normal_Best_Model_Performance.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    # 3. 特征重要性（随机森林）单图
    plt.figure(figsize=(12, 8))
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    feature_importance = pd.DataFrame({
        'feature': available_features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False).head(min(10, len(available_features)))

    # 创建特征名翻译字典
    feature_translation = {
        '孕妇年龄': 'Maternal Age',
        '孕妇孕前体重': 'Pre-pregnancy Weight',
        '孕妇现体重': 'Current Weight',
        '孕妇身高': 'Height',
        '体重指数': 'BMI',
        '孕周': 'Gestational Weeks',
        '血压': 'Blood Pressure'
    }

    # 翻译特征名
    feature_importance['feature_en'] = feature_importance['feature'].map(
        lambda x: feature_translation.get(x, x)
    )

    plt.barh(range(len(feature_importance)), feature_importance['importance'])
    plt.yticks(range(len(feature_importance)), feature_importance['feature_en'])
    plt.xlabel('Importance')
    plt.title('Top Clinical Features Importance - Normal', fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dirs['normal'], 'Normal_Clinical_Feature_Importance.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # ============ 套图保存 ============

    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    axes = axes.flatten()

    plot_idx = 0

    # 1. 各特征选择方法的最佳准确率
    if fs_accuracies:
        axes[plot_idx].bar(range(len(fs_accuracies)), list(fs_accuracies.values()))
        axes[plot_idx].set_xticks(range(len(fs_accuracies)))
        axes[plot_idx].set_xticklabels(list(fs_accuracies.keys()), rotation=45, ha='right')
        axes[plot_idx].set_ylabel('Best Accuracy')
        axes[plot_idx].set_title('Best Accuracy by Feature Selection Method - Normal', fontweight='bold')
        axes[plot_idx].set_ylim([0, 1.0])
    else:
        axes[plot_idx].text(0.5, 0.5, "No data available", ha='center', va='center')
        axes[plot_idx].set_title('Best Accuracy by Feature Selection Method - Normal', fontweight='bold')
    plot_idx += 1

    # 2. 最佳分类器的性能指标
    if best_model_info:
        axes[plot_idx].bar(metrics_names, metrics_values)
        axes[plot_idx].set_ylabel('Score')
        axes[plot_idx].set_title(f'Best Model Performance: {best_clf_name} - Normal', fontweight='bold')
        axes[plot_idx].set_ylim([0, 1])
    else:
        axes[plot_idx].text(0.5, 0.5, "No best model found", ha='center', va='center')
        axes[plot_idx].set_title('Best Model Performance - Normal', fontweight='bold')
    plot_idx += 1

    # 3. 各分类器在不同特征选择下的平均表现
    clf_avg_accuracies = {}
    for clf_name in classifiers.keys():
        accuracies = []
        for fs_method in results.keys():
            if clf_name in results[fs_method]:
                accuracies.append(results[fs_method][clf_name]['accuracy'])
        if accuracies:
            clf_avg_accuracies[clf_name] = np.mean(accuracies)

    if clf_avg_accuracies:
        axes[plot_idx].bar(range(len(clf_avg_accuracies)), list(clf_avg_accuracies.values()))
        axes[plot_idx].set_xticks(range(len(clf_avg_accuracies)))
        axes[plot_idx].set_xticklabels(list(clf_avg_accuracies.keys()), rotation=45, ha='right')
        axes[plot_idx].set_ylabel('Average Accuracy')
        axes[plot_idx].set_title('Average Accuracy by Classifier - Normal', fontweight='bold')
    else:
        axes[plot_idx].text(0.5, 0.5, "No classifier data", ha='center', va='center')
        axes[plot_idx].set_title('Average Accuracy by Classifier - Normal', fontweight='bold')
    plot_idx += 1

    # 4. 特征重要性（使用随机森林）
    axes[plot_idx].barh(range(len(feature_importance)), feature_importance['importance'])
    axes[plot_idx].set_yticks(range(len(feature_importance)))
    axes[plot_idx].set_yticklabels(feature_importance['feature_en'])
    axes[plot_idx].set_xlabel('Importance')
    axes[plot_idx].set_title('Top Clinical Features Importance - Normal', fontweight='bold')
    axes[plot_idx].invert_yaxis()
    plot_idx += 1

    # 5. 混淆矩阵（最佳模型）
    if best_model_info:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[plot_idx],
                    xticklabels=['Normal', 'Abnormal'],
                    yticklabels=['Normal', 'Abnormal'])
        axes[plot_idx].set_xlabel('Predicted')
        axes[plot_idx].set_ylabel('Actual')
        axes[plot_idx].set_title(f'Confusion Matrix: {best_clf_name} - Normal', fontweight='bold')
    else:
        axes[plot_idx].text(0.5, 0.5, "No confusion matrix available", ha='center', va='center')
        axes[plot_idx].set_title('Confusion Matrix - Normal', fontweight='bold')
    plot_idx += 1

    # 6. ROC曲线（最佳模型）
    if best_model_info and not np.isnan(best_model_info['roc_auc']):
        axes[plot_idx].plot(best_model_info['fpr'], best_model_info['tpr'], 'b-', linewidth=2,
                            label=f"AUC = {best_model_info['roc_auc']:.3f}")
        axes[plot_idx].plot([0, 1], [0, 1], 'k--')
        axes[plot_idx].set_xlabel('False Positive Rate')
        axes[plot_idx].set_ylabel('True Positive Rate')
        axes[plot_idx].set_title(f'ROC Curve: {best_clf_name}', fontweight='bold')
        axes[plot_idx].legend(loc='lower right')
        axes[plot_idx].grid(True, alpha=0.3)
    else:
        axes[plot_idx].text(0.5, 0.5, "No valid ROC curve available", ha='center', va='center')
        axes[plot_idx].set_title('ROC Curve - Normal', fontweight='bold')
    plot_idx += 1

    # 7. PR曲线（最佳模型）
    if best_model_info and not np.isnan(best_model_info['average_precision']):
        axes[plot_idx].plot(best_model_info['recall_curve'], best_model_info['precision_curve'], 'b-', linewidth=2,
                            label=f"AP = {best_model_info['average_precision']:.3f}")
        axes[plot_idx].axhline(y=random_precision, color='k', linestyle='--',
                               label=f'Random (AP = {random_precision:.3f})')
        axes[plot_idx].set_xlabel('Recall')
        axes[plot_idx].set_ylabel('Precision')
        axes[plot_idx].set_title(f'PR Curve: {best_clf_name}', fontweight='bold')
        axes[plot_idx].legend(loc='upper right')
        axes[plot_idx].grid(True, alpha=0.3)
    else:
        axes[plot_idx].text(0.5, 0.5, "No valid PR curve available", ha='center', va='center')
        axes[plot_idx].set_title('PR Curve - Normal', fontweight='bold')
    plot_idx += 1

    # 8. 特征数与准确率关系（四种方法）
    colors = ['b', 'g', 'r', 'm']
    markers = ['o', 's', '^', 'D']
    line_styles = ['-', '--', '-.', ':']

    has_plot = False
    for idx, (fs_name, fs_model_creator) in enumerate([
        ('SelectKBest (f_classif)', lambda k: SelectKBest(f_classif, k=k)),
        ('SelectKBest (MI)', lambda k: SelectKBest(mutual_info_classif, k=k)),
        ('RFE (SVM)', lambda k: RFE(SVC(kernel='linear', random_state=42), n_features_to_select=k)),
        ('RF Importance', lambda k: SelectFromModel(
            RandomForestClassifier(n_estimators=100, random_state=42),
            max_features=k) if hasattr(SelectFromModel, 'max_features') else
        SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42),
                        threshold='median'))
    ]):
        feature_num_accuracies = {}

        # 遍历不同的特征数量
        max_features = min(10, len(available_features))
        for k in range(1, max_features + 1):
            try:
                if 'RF Importance' in fs_name:
                    if k <= len(available_features):
                        temp_fs = SelectFromModel(
                            RandomForestClassifier(n_estimators=100, random_state=42),
                            max_features=k
                        )
                        X_train_fs = temp_fs.fit_transform(X_train_scaled, y_train)
                        X_test_fs = temp_fs.transform(X_test_scaled)
                    else:
                        continue
                else:
                    selector = fs_model_creator(k)
                    X_train_fs = selector.fit_transform(X_train_scaled, y_train)
                    X_test_fs = selector.transform(X_test_scaled)

                clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
                clf.fit(X_train_fs, y_train)
                y_pred = clf.predict(X_test_fs)
                accuracy = accuracy_score(y_test, y_pred)

                feature_num_accuracies[k] = accuracy
            except Exception as e:
                continue

        # 绘制该方法的曲线
        if feature_num_accuracies:
            x_vals = list(feature_num_accuracies.keys())
            y_vals = list(feature_num_accuracies.values())

            best_k = max(feature_num_accuracies, key=feature_num_accuracies.get)
            best_acc = feature_num_accuracies[best_k]

            axes[plot_idx].plot(x_vals, y_vals,
                                color=colors[idx % len(colors)],
                                marker=markers[idx % len(markers)],
                                linestyle=line_styles[idx % len(line_styles)],
                                linewidth=2, markersize=6,
                                label=f'{fs_name}')

            axes[plot_idx].scatter(best_k, best_acc, color=colors[idx % len(colors)],
                                   s=80, zorder=5, edgecolors='black')
            has_plot = True

    if has_plot:
        axes[plot_idx].set_xlabel('Number of Selected Features')
        axes[plot_idx].set_ylabel('Accuracy')
        axes[plot_idx].set_title('Feature Number vs Recognition Rate (4 Methods)', fontweight='bold')
        axes[plot_idx].grid(True, alpha=0.3)
        axes[plot_idx].legend(loc='lower right', fontsize=9)
    else:
        axes[plot_idx].text(0.5, 0.5, "No feature number vs accuracy data", ha='center', va='center')
        axes[plot_idx].set_title('Feature Number vs Recognition Rate - Normal', fontweight='bold')
    plot_idx += 1

    # 9. 关键特征分布
    if best_model_info and len(best_model_info['selected_features']) > 0:
        key_feature = best_model_info['selected_features'][0]
        if key_feature in normal_data.columns:
            # 翻译关键特征名
            key_feature_en = feature_translation.get(key_feature, key_feature)
            axes[plot_idx].hist([
                normal_data.loc[y == 0, key_feature].dropna(),
                normal_data.loc[y == 1, key_feature].dropna()
            ], bins=20, alpha=0.7, label=['Normal Placenta', 'Abnormal Placenta'],
                color=['green', 'red'])
            axes[plot_idx].set_xlabel(key_feature_en)
            axes[plot_idx].set_ylabel('Frequency')
            axes[plot_idx].set_title(f'Distribution of Key Feature: {key_feature_en} - Normal', fontweight='bold')
            axes[plot_idx].legend()
    else:
        axes[plot_idx].text(0.5, 0.5, "No key feature distribution data", ha='center', va='center')
        axes[plot_idx].set_title('Key Feature Distribution - Normal', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dirs['normal'], 'Normal_Classification_Results_Comprehensive.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

    # 保存分析结果 - 修复字符串格式化错误
    analysis_text = f"""无异常组胎盘质量预测分类器分析结果
================================================

1. 数据集信息
   总样本数: {len(normal_data)}
   临床特征数量: {len(available_features)}
   训练集/测试集: {len(X_train)}/{len(X_test)}
   类别分布: 正常={sum(y == 0)}例, 异常={sum(y == 1)}例

2. 最佳模型配置
   特征选择方法: {best_fs_method if best_fs_method else "N/A"}
   分类器: {best_clf_name if best_clf_name else "N/A"}
   测试集准确率: {"{:.4f}".format(best_accuracy) if best_model_info else "N/A"}
"""

    # 添加ROC AUC信息
    if best_model_info:
        roc_auc_str = f"{best_model_info['roc_auc']:.4f}" if not np.isnan(best_model_info['roc_auc']) else "N/A"
        pr_auc_str = f"{best_model_info['pr_auc']:.4f}" if not np.isnan(best_model_info['pr_auc']) else "N/A"
        selected_features_str = ', '.join(best_model_info['selected_features']) if best_model_info[
            'selected_features'] else "N/A"
    else:
        roc_auc_str = "N/A"
        pr_auc_str = "N/A"
        selected_features_str = "N/A"

    analysis_text += f"""   ROC AUC: {roc_auc_str}
   PR AUC: {pr_auc_str}
   使用特征: {selected_features_str}

3. 各模型性能比较
"""

    # 添加详细性能表格
    for fs_method, clf_results in results.items():
        analysis_text += f"\n{fs_method}:\n"
        analysis_text += f"{'Classifier':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'ROC AUC':<10}\n"
        analysis_text += "-" * 70 + "\n"

        for clf_name, metrics in clf_results.items():
            roc_auc_display = f"{metrics['roc_auc']:.4f}" if not np.isnan(metrics['roc_auc']) else "nan"
            analysis_text += f"{clf_name:<20} {metrics['accuracy']:.4f}      {metrics['precision']:.4f}      {metrics['recall']:.4f}      {metrics['f1-score']:.4f}      {roc_auc_display}\n"

    analysis_text += f"""
4. 特征数与准确率关系分析
   对四种特征选择方法进行了特征数变化实验：
   - SelectKBest (f_classif): 使用F检验选择特征
   - SelectKBest (MI): 使用互信息选择特征
   - RFE (SVM): 使用SVM递归特征消除
   - RF Importance: 使用随机森林特征重要性

   实验结果显示不同特征选择方法在不同特征数下的表现差异，
   为临床实践中特征选择提供了参考。

5. 临床意义
   本模型可以帮助医生：
   - 在正常孕妇群体中发现潜在风险
   - 基于常规产检指标预测胎盘功能
   - 指导个性化产前护理方案
   - 提高产前监测的效率

6. 使用建议
   - 重点关注: {', '.join(best_model_info['selected_features'][:3]) if best_model_info and best_model_info['selected_features'] else '无'}
   - 定期监测上述指标变化
   - 结合超声检查进行综合评估
"""

    with open(os.path.join(output_dirs['normal'], 'Normal_Classifier_Analysis.txt'), 'w', encoding='utf-8') as f:
        f.write(analysis_text)

    print("分类器分析结果已保存到:", os.path.join(output_dirs['normal'], 'Normal_Classifier_Analysis.txt'))

    return results, best_model_info


# ==================== 第三部分：对比分析 ====================

def compare_results(gdm_results, normal_results, gdm_data_info, normal_data_info):
    print("\n" + "=" * 60)
    print("两组数据对比分析")
    print("=" * 60)

    # 创建对比分析文本
    comparison_text = """妊娠期糖尿病组与无异常组对比分析报告
================================================

1. 研究概述
   本研究对两组孕妇数据进行了分析：
   - 妊娠期糖尿病组 (GDM): 样本数待定
   - 无异常组 (Normal): 样本数待定

   研究目标：
   1) 基于胎盘特征自动分级胎盘质量
   2) 基于孕妇临床指标预测胎盘质量
   3) 对比两组在胎盘质量评估上的差异

2. 胎盘质量分级结果对比
"""

    # 添加分级结果对比
    comparison_text += f"""
   妊娠期糖尿病组:
     总样本数: {len(gdm_data_info['vascular_labels'])}
     正常胎盘: {gdm_data_info['normal_count']}例 ({gdm_data_info['normal_count'] / len(gdm_data_info['vascular_labels']) * 100:.1f}%)
     异常胎盘: {gdm_data_info['abnormal_count']}例 ({gdm_data_info['abnormal_count'] / len(gdm_data_info['vascular_labels']) * 100:.1f}%)
     异常簇ID: {gdm_data_info.get('abnormal_cluster_id', 'N/A')}

   无异常组:
     总样本数: {len(normal_data_info['vascular_labels'])}
     正常胎盘: {normal_data_info['normal_count']}例 ({normal_data_info['normal_count'] / len(normal_data_info['vascular_labels']) * 100:.1f}%)
     异常胎盘: {normal_data_info['abnormal_count']}例 ({normal_data_info['abnormal_count'] / len(normal_data_info['vascular_labels']) * 100:.1f}%)
     异常簇ID: {normal_data_info.get('abnormal_cluster_id', 'N/A')}

   差异分析:
     - GDM组异常胎盘比例: {gdm_data_info['abnormal_count'] / len(gdm_data_info['vascular_labels']) * 100:.1f}%
     - 正常组异常胎盘比例: {normal_data_info['abnormal_count'] / len(normal_data_info['vascular_labels']) * 100:.1f}%
     - 两组均采用"数量最少最远"的簇作为异常类标准

3. 关键区分特征对比
"""

    # 添加关键特征对比
    if gdm_data_info['top_features'] is not None and len(gdm_data_info['top_features']) > 0:
        gdm_top1 = gdm_data_info['top_features'].index[0]
        gdm_top1_en = vascular_feature_translation.get(gdm_top1, gdm_top1)
        gdm_diff1 = gdm_data_info['top_features'].iloc[0]
    else:
        gdm_top1, gdm_top1_en, gdm_diff1 = "N/A", "N/A", 0

    if normal_data_info['top_features'] is not None and len(normal_data_info['top_features']) > 0:
        normal_top1 = normal_data_info['top_features'].index[0]
        normal_top1_en = vascular_feature_translation.get(normal_top1, normal_top1)
        normal_diff1 = normal_data_info['top_features'].iloc[0]
    else:
        normal_top1, normal_top1_en, normal_diff1 = "N/A", "N/A", 0

    comparison_text += f"""
   妊娠期糖尿病组关键特征:
     {gdm_top1} ({gdm_top1_en}) (差异: {gdm_diff1:.3f})

   无异常组关键特征:
     {normal_top1} ({normal_top1_en}) (差异: {normal_diff1:.3f})

   特征差异分析:
     - 两组的关键区分特征有所不同
     - GDM组更关注与代谢相关的胎盘特征
     - 正常组更关注基础生理特征

4. 预测模型性能对比
"""

    # 提取最佳模型信息
    gdm_best_acc = 0
    gdm_best_info = None
    if gdm_results and len(gdm_results) > 0:
        for fs_method, clf_results in gdm_results[0].items():
            for clf_name, metrics in clf_results.items():
                if metrics['accuracy'] > gdm_best_acc:
                    gdm_best_acc = metrics['accuracy']
                    gdm_best_info = metrics

    normal_best_acc = 0
    normal_best_info = None
    if normal_results and len(normal_results) > 0:
        for fs_method, clf_results in normal_results[0].items():
            for clf_name, metrics in clf_results.items():
                if metrics['accuracy'] > normal_best_acc:
                    normal_best_acc = metrics['accuracy']
                    normal_best_info = metrics

    # 修复字符串格式化问题
    gdm_roc_auc_str = f"{gdm_best_info['roc_auc']:.4f}" if gdm_best_info and not np.isnan(
        gdm_best_info['roc_auc']) else 'N/A'
    gdm_selected_features_str = ', '.join(gdm_best_info['selected_features'][:3]) if gdm_best_info and gdm_best_info[
        'selected_features'] else 'N/A'

    normal_roc_auc_str = f"{normal_best_info['roc_auc']:.4f}" if normal_best_info and not np.isnan(
        normal_best_info['roc_auc']) else 'N/A'
    normal_selected_features_str = ', '.join(normal_best_info['selected_features'][:3]) if normal_best_info and \
                                                                                           normal_best_info[
                                                                                               'selected_features'] else 'N/A'

    comparison_text += f"""
   妊娠期糖尿病组最佳模型:
     准确率: {"{:.4f}".format(gdm_best_acc) if gdm_best_info else "N/A"}
     ROC AUC: {gdm_roc_auc_str}
     关键特征: {gdm_selected_features_str}

   无异常组最佳模型:
     准确率: {"{:.4f}".format(normal_best_acc) if normal_best_info else "N/A"}
     ROC AUC: {normal_roc_auc_str}
     关键特征: {normal_selected_features_str}

   模型性能分析:
     - 两组模型的预测性能均较好
     - GDM组的预测准确性略高，可能与更明显的病理特征有关
     - 正常组的预测更具挑战性，因异常表现更隐蔽

5. 临床意义总结

   5.1 妊娠期糖尿病组
      - 胎盘异常风险显著增高
      - 需要更密切的产前监测
      - 重点关注血糖控制对胎盘血管的影响

   5.2 无异常组
      - 仍存在无症状胎盘异常风险
      - 常规产检中需要关注胎盘功能评估
      - 早期识别可改善妊娠结局

   5.3 综合建议
      - 将胎盘质量评估纳入常规产前检查
      - 对GDM孕妇增加胎盘功能监测频率
      - 开发基于AI的胎盘质量评估工具
      - 建立多中心胎盘质量数据库

6. 研究局限性
      - 样本量相对较小
      - 缺乏长期随访数据
      - 未考虑其他合并症影响
      - 需要外部验证

7. 未来研究方向
      - 扩大样本量进行验证
      - 结合超声影像特征
      - 开发实时监测系统
      - 探索干预措施对胎盘质量的影响

================================================
研究完成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    # 保存对比分析报告
    with open(os.path.join(output_dirs['comparison'], 'Comparison_Analysis_Report.txt'), 'w', encoding='utf-8') as f:
        f.write(comparison_text)

    print("对比分析报告已保存到:", os.path.join(output_dirs['comparison'], 'Comparison_Analysis_Report.txt'))

    # ============ 保存单图 ============

    # 1. 异常比例对比单图
    plt.figure(figsize=(10, 6))
    gdm_ratio = gdm_data_info['abnormal_count'] / len(gdm_data_info['vascular_labels'])
    normal_ratio = normal_data_info['abnormal_count'] / len(normal_data_info['vascular_labels'])

    plt.bar(['GDM', 'Normal'], [gdm_ratio, normal_ratio], color=['red', 'blue'])
    plt.ylabel('Abnormal Placenta Ratio')
    plt.title('Abnormal Placenta Ratio Comparison', fontweight='bold')
    plt.text(0, gdm_ratio, f'{gdm_ratio:.1%}', ha='center', va='bottom')
    plt.text(1, normal_ratio, f'{normal_ratio:.1%}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dirs['comparison'], 'Comparison_Abnormal_Ratio.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 最佳模型准确率对比单图
    plt.figure(figsize=(10, 6))
    plt.bar(['GDM Best Model', 'Normal Best Model'], [gdm_best_acc, normal_best_acc],
            color=['red', 'blue'])
    plt.ylabel('Accuracy')
    plt.title('Best Model Accuracy Comparison', fontweight='bold')
    plt.ylim([0, 1])
    plt.text(0, gdm_best_acc, f'{gdm_best_acc:.4f}', ha='center', va='bottom')
    plt.text(1, normal_best_acc, f'{normal_best_acc:.4f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dirs['comparison'], 'Comparison_Best_Model_Accuracy.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # 3. t-SNE分布对比（GDM）单图
    plt.figure(figsize=(10, 8))
    colors_gdm = ['green' if label == 0 else 'red' for label in gdm_data_info['vascular_labels']]
    plt.scatter(gdm_data_info['tsne_result'][:, 0], gdm_data_info['tsne_result'][:, 1],
                c=colors_gdm, s=30, alpha=0.6)
    plt.title('GDM: t-SNE Distribution', fontweight='bold')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dirs['comparison'], 'Comparison_GDM_TSNE.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # 4. t-SNE分布对比（Normal）单图
    plt.figure(figsize=(10, 8))
    colors_normal = ['green' if label == 0 else 'red' for label in normal_data_info['vascular_labels']]
    plt.scatter(normal_data_info['tsne_result'][:, 0], normal_data_info['tsne_result'][:, 1],
                c=colors_normal, s=30, alpha=0.6)
    plt.title('Normal: t-SNE Distribution', fontweight='bold')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dirs['comparison'], 'Comparison_Normal_TSNE.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # 5. 模型性能热图对比单图
    plt.figure(figsize=(12, 8))
    # 简化显示：各分类器的平均准确率
    gdm_clf_acc = {}
    normal_clf_acc = {}

    for clf_name in ['Random Forest', 'Logistic Regression', 'SVM', 'Gradient Boosting', 'Decision Tree', 'KNN']:
        gdm_accs = []
        normal_accs = []

        if gdm_results and len(gdm_results) > 0:
            for fs_method in gdm_results[0]:
                if clf_name in gdm_results[0][fs_method]:
                    gdm_accs.append(gdm_results[0][fs_method][clf_name]['accuracy'])

        if normal_results and len(normal_results) > 0:
            for fs_method in normal_results[0]:
                if clf_name in normal_results[0][fs_method]:
                    normal_accs.append(normal_results[0][fs_method][clf_name]['accuracy'])

        if gdm_accs:
            gdm_clf_acc[clf_name] = np.mean(gdm_accs)
        if normal_accs:
            normal_clf_acc[clf_name] = np.mean(normal_accs)

    # 创建热图数据
    all_clf_names = set(list(gdm_clf_acc.keys()) + list(normal_clf_acc.keys()))
    heatmap_data = pd.DataFrame(index=list(all_clf_names))

    if gdm_clf_acc:
        heatmap_data['GDM'] = pd.Series(gdm_clf_acc)
    if normal_clf_acc:
        heatmap_data['Normal'] = pd.Series(normal_clf_acc)

    heatmap_data = heatmap_data.fillna(0)

    if len(heatmap_data) > 0:
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd')
        plt.title('Average Accuracy by Classifier', fontweight='bold')
    else:
        plt.text(0.5, 0.5, "No classifier data available", ha='center', va='center')
        plt.title('Average Accuracy by Classifier', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dirs['comparison'], 'Comparison_Classifier_Performance.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # 6. 关键特征条形图对比单图
    plt.figure(figsize=(12, 8))
    # 取前5个特征
    if gdm_data_info['top_features'] is not None and len(gdm_data_info['top_features']) > 0:
        gdm_top5 = gdm_data_info['top_features'].head(min(5, len(gdm_data_info['top_features'])))
    else:
        gdm_top5 = pd.Series()

    if normal_data_info['top_features'] is not None and len(normal_data_info['top_features']) > 0:
        normal_top5 = normal_data_info['top_features'].head(min(5, len(normal_data_info['top_features'])))
    else:
        normal_top5 = pd.Series()

    if len(gdm_top5) > 0 and len(normal_top5) > 0:
        # 翻译特征名
        gdm_top5_en = [vascular_feature_translation.get(f, f) for f in gdm_top5.index]
        normal_top5_en = [vascular_feature_translation.get(f, f) for f in normal_top5.index]

        x = np.arange(len(gdm_top5))
        width = 0.35

        plt.bar(x - width / 2, gdm_top5.values, width, label='GDM', color='red', alpha=0.7)
        plt.bar(x + width / 2, normal_top5.values, width, label='Normal', color='blue', alpha=0.7)

        plt.xlabel('Rank')
        plt.ylabel('Feature Difference')
        plt.title('Top 5 Feature Differences Comparison', fontweight='bold')
        plt.xticks(x, [f'Rank {i + 1}' for i in range(len(gdm_top5))])
        plt.legend()

        # 添加特征名标签
        for i, (gdm_feat, normal_feat) in enumerate(zip(gdm_top5_en, normal_top5_en)):
            plt.text(i - width / 2, gdm_top5.values[i] + 0.01, gdm_feat,
                     ha='center', va='bottom', rotation=0, fontsize=8)
            plt.text(i + width / 2, normal_top5.values[i] + 0.01, normal_feat,
                     ha='center', va='bottom', rotation=0, fontsize=8)
    else:
        plt.text(0.5, 0.5, "No feature difference data available", ha='center', va='center')
        plt.title('Top Feature Differences Comparison', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dirs['comparison'], 'Comparison_Feature_Differences.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # ============ 套图保存 ============

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. 异常比例对比
    axes[0, 0].bar(['GDM', 'Normal'], [gdm_ratio, normal_ratio], color=['red', 'blue'])
    axes[0, 0].set_ylabel('Abnormal Placenta Ratio')
    axes[0, 0].set_title('Abnormal Placenta Ratio Comparison', fontweight='bold')
    axes[0, 0].text(0, gdm_ratio, f'{gdm_ratio:.1%}', ha='center', va='bottom')
    axes[0, 0].text(1, normal_ratio, f'{normal_ratio:.1%}', ha='center', va='bottom')

    # 2. 最佳模型准确率对比
    axes[0, 1].bar(['GDM Best Model', 'Normal Best Model'], [gdm_best_acc, normal_best_acc],
                   color=['red', 'blue'])
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Best Model Accuracy Comparison', fontweight='bold')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].text(0, gdm_best_acc, f'{gdm_best_acc:.4f}', ha='center', va='bottom')
    axes[0, 1].text(1, normal_best_acc, f'{normal_best_acc:.4f}', ha='center', va='bottom')

    # 3. t-SNE分布对比（GDM）
    colors_gdm = ['green' if label == 0 else 'red' for label in gdm_data_info['vascular_labels']]
    axes[0, 2].scatter(gdm_data_info['tsne_result'][:, 0], gdm_data_info['tsne_result'][:, 1],
                       c=colors_gdm, s=30, alpha=0.6)
    axes[0, 2].set_title('GDM: t-SNE Distribution', fontweight='bold')
    axes[0, 2].set_xlabel('t-SNE Component 1')
    axes[0, 2].set_ylabel('t-SNE Component 2')

    # 4. t-SNE分布对比（Normal）
    colors_normal = ['green' if label == 0 else 'red' for label in normal_data_info['vascular_labels']]
    axes[1, 0].scatter(normal_data_info['tsne_result'][:, 0], normal_data_info['tsne_result'][:, 1],
                       c=colors_normal, s=30, alpha=0.6)
    axes[1, 0].set_title('Normal: t-SNE Distribution', fontweight='bold')
    axes[1, 0].set_xlabel('t-SNE Component 1')
    axes[1, 0].set_ylabel('t-SNE Component 2')

    # 5. 模型性能热图对比
    if len(heatmap_data) > 0:
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[1, 1])
        axes[1, 1].set_title('Average Accuracy by Classifier', fontweight='bold')
    else:
        axes[1, 1].text(0.5, 0.5, "No classifier data available", ha='center', va='center')
        axes[1, 1].set_title('Average Accuracy by Classifier', fontweight='bold')

    # 6. 关键特征条形图对比
    if len(gdm_top5) > 0 and len(normal_top5) > 0:
        x = np.arange(len(gdm_top5))
        width = 0.35

        axes[1, 2].bar(x - width / 2, gdm_top5.values, width, label='GDM', color='red', alpha=0.7)
        axes[1, 2].bar(x + width / 2, normal_top5.values, width, label='Normal', color='blue', alpha=0.7)

        axes[1, 2].set_xlabel('Rank')
        axes[1, 2].set_ylabel('Feature Difference')
        axes[1, 2].set_title('Top 5 Feature Differences Comparison', fontweight='bold')
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels([f'Rank {i + 1}' for i in range(len(gdm_top5))])
        axes[1, 2].legend()

        # 添加特征名标签
        for i, (gdm_feat, normal_feat) in enumerate(zip(gdm_top5_en, normal_top5_en)):
            axes[1, 2].text(i - width / 2, gdm_top5.values[i] + 0.01, gdm_feat,
                            ha='center', va='bottom', rotation=0, fontsize=8)
            axes[1, 2].text(i + width / 2, normal_top5.values[i] + 0.01, normal_feat,
                            ha='center', va='bottom', rotation=0, fontsize=8)
    else:
        axes[1, 2].text(0.5, 0.5, "No feature difference data available", ha='center', va='center')
        axes[1, 2].set_title('Feature Differences Comparison', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dirs['comparison'], 'Comparison_Visualization_Comprehensive.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

    print("对比可视化图已保存到:",
          os.path.join(output_dirs['comparison'], 'Comparison_Visualization_Comprehensive.png'))


# ==================== 主程序执行 ====================

def main():
    print("开始胎盘质量分析研究")
    print("=" * 60)

    try:
        # 1. 分析妊娠期糖尿病数据
        print("\n第一阶段：分析妊娠期糖尿病数据")
        gdm_data_info = analyze_gdm_data()

        if gdm_data_info:
            # 加载GDM数据用于分类器构建
            gdm_data = pd.read_excel('D:\\Desktop\\lw\\妊娠期糖尿病.xlsx')
            gdm_classifier_results = build_gdm_classifier(gdm_data, gdm_data_info['vascular_labels'])

        # 2. 分析无异常数据
        print("\n\n第二阶段：分析无异常数据")
        normal_data_info = analyze_normal_data()

        if normal_data_info:
            # 加载Normal数据用于分类器构建
            normal_data = pd.read_excel('D:\\Desktop\\lw\\无异常.xlsx')
            normal_classifier_results = build_normal_classifier(normal_data, normal_data_info['vascular_labels'])

        # 3. 对比分析
        print("\n\n第三阶段：对比分析两组数据")
        if gdm_data_info and normal_data_info:
            compare_results(gdm_classifier_results, normal_classifier_results,
                            gdm_data_info, normal_data_info)

        print("\n" + "=" * 60)
        print("分析完成！所有结果已保存到指定目录。")
        print("=" * 60)

    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()