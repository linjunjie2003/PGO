import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu
import warnings

warnings.filterwarnings('ignore')


def convert_gestational_week(week_str):
    """
    将孕周字符串转换为数值（周）
    例如: "38w+6" -> 38.86 (38 + 6/7)
    """
    if pd.isna(week_str):
        return np.nan

    week_str = str(week_str).strip()
    week_str = week_str.replace('w', '').replace('W', '')

    if '+' in week_str:
        weeks, days = week_str.split('+')
        weeks = float(weeks) if weeks else 0
        days = float(days) if days else 0
        return weeks + days / 7
    else:
        try:
            return float(week_str)
        except:
            return np.nan


def clean_numeric_data(data):
    """
    清理数据，将非数值数据转换为NaN
    """
    if isinstance(data, (int, float)):
        return float(data)
    elif isinstance(data, str):
        try:
            # 尝试转换为浮点数
            return float(data)
        except ValueError:
            # 如果是"无"或其他非数值字符串，返回NaN
            if data.strip() == "无":
                return np.nan
            else:
                # 尝试提取数字
                import re
                numbers = re.findall(r'\d+\.?\d*', data)
                if numbers:
                    return float(numbers[0])
                else:
                    return np.nan
    else:
        return np.nan


def process_data(gdm_path, normal_path):
    """
    处理GDM组和非糖尿病组数据
    """
    # 读取GDM组数据
    gdm_df = pd.read_excel(gdm_path)

    # 读取非糖尿病组数据
    normal_df = pd.read_excel(normal_path)

    # 清理数值数据
    numeric_columns = ['孕妇年龄', '孕妇孕前体重', '孕妇现体重', '孕妇身高',
                       '空腹血糖孕7-9月', '糖化血红蛋白孕7-9月', '胎盘重量']

    for col in numeric_columns:
        if col in gdm_df.columns:
            gdm_df[col] = gdm_df[col].apply(clean_numeric_data)
        if col in normal_df.columns:
            normal_df[col] = normal_df[col].apply(clean_numeric_data)

    # 转换孕周数据
    gdm_df['孕周'] = gdm_df['孕周'].apply(convert_gestational_week)
    normal_df['孕周'] = normal_df['孕周'].apply(convert_gestational_week)

    # 计算体重指数（如果未提供）
    if '体重指数' not in gdm_df.columns:
        gdm_df['体重指数'] = gdm_df['孕妇孕前体重'] / (gdm_df['孕妇身高'] / 100) ** 2

    if '体重指数' not in normal_df.columns:
        normal_df['体重指数'] = normal_df['孕妇孕前体重'] / (normal_df['孕妇身高'] / 100) ** 2

    # 清理体重指数中的NaN
    gdm_df['体重指数'] = gdm_df['体重指数'].apply(clean_numeric_data)
    normal_df['体重指数'] = normal_df['体重指数'].apply(clean_numeric_data)

    return gdm_df, normal_df


def check_normality(data):
    """
    检查数据是否服从正态分布
    """
    # 移除NaN值
    clean_data = data.dropna()

    if len(clean_data) < 3:
        return False

    try:
        # Shapiro-Wilk检验
        stat, p = stats.shapiro(clean_data)
        return p > 0.05
    except:
        # 如果检验失败，返回False
        return False


def calculate_statistics(data, variable_name):
    """
    计算变量的统计量
    修改：无论分布如何，都返回均值±标准差
    """
    # 移除缺失值
    clean_data = data.dropna()

    if len(clean_data) == 0:
        return "N/A"

    # 计算均值和标准差
    mean = np.mean(clean_data)
    std = np.std(clean_data, ddof=1)  # 使用样本标准差

    return f"{mean:.2f}±{std:.2f}"


def calculate_p_value(gdm_data, normal_data):
    """
    计算两组之间的p值
    """
    # 移除缺失值
    gdm_clean = gdm_data.dropna()
    normal_clean = normal_data.dropna()

    if len(gdm_clean) < 2 or len(normal_clean) < 2:
        return "N/A"

    try:
        # 检查正态性
        gdm_normal = check_normality(gdm_clean)
        normal_normal = check_normality(normal_clean)

        if gdm_normal and normal_normal:
            # 两独立样本t检验
            t_stat, p_value = ttest_ind(gdm_clean, normal_clean, equal_var=False)
        else:
            # Mann-Whitney U检验
            u_stat, p_value = mannwhitneyu(gdm_clean, normal_clean, alternative='two-sided')

        # 格式化p值
        if p_value < 0.001:
            return "<0.001"
        else:
            return f"{p_value:.3f}"
    except:
        return "N/A"


def generate_table1(gdm_df, normal_df):
    """
    生成表1: 临床特征比较
    """
    # 定义要分析的变量及其英文名称
    variables = {
        '孕妇年龄': 'Maternal age (years)',
        '孕妇孕前体重': 'Pre-pregnancy weight (kg)',
        '孕妇现体重': 'Current weight (kg)',
        '孕妇身高': 'Height (cm)',
        '体重指数': 'Body mass index (kg/m²)',
        '孕周': 'Gestational age (weeks)',
        '胎盘重量': 'Placental weight (kg)'
    }

    # GDM组特有的变量
    gdm_specific_vars = {
        '空腹血糖孕7-9月': 'Fasting blood glucose (mmol/L)',
        '糖化血红蛋白孕7-9月': 'Glycated hemoglobin (%)'
    }

    # 创建结果表格
    results = []

    # 处理共同变量
    for var, eng_name in variables.items():
        gdm_stat = calculate_statistics(gdm_df[var], var)
        normal_stat = calculate_statistics(normal_df[var], var)
        p_value = calculate_p_value(gdm_df[var], normal_df[var])

        results.append({
            'Characteristic': eng_name,
            'GDM group (n=42)': gdm_stat,
            'Non-diabetic group (n=124)': normal_stat,
            'p-value': p_value
        })

    # 处理GDM组特有变量
    for var, eng_name in gdm_specific_vars.items():
        if var in gdm_df.columns:
            gdm_stat = calculate_statistics(gdm_df[var], var)

            results.append({
                'Characteristic': eng_name,
                'GDM group (n=42)': gdm_stat,
                'Non-diabetic group (n=124)': '-',
                'p-value': '-'
            })

    # 创建DataFrame
    table1_df = pd.DataFrame(results)

    return table1_df


def main():
    # 文件路径
    gdm_path = r"D:\Desktop\lw\1.xlsx"
    normal_path = r"D:\Desktop\lw\2.xlsx"

    try:
        # 处理数据
        print("正在处理数据...")
        gdm_df, normal_df = process_data(gdm_path, normal_path)

        # 打印基本信息
        print(f"GDM组样本数: {len(gdm_df)}")
        print(f"非糖尿病组样本数: {len(normal_df)}")

        # 显示数据概览
        print("\n" + "=" * 80)
        print("数据概览:")
        print("=" * 80)
        print("\nGDM组数据列:", gdm_df.columns.tolist())
        print("\nGDM组缺失值统计:")
        print(gdm_df.isnull().sum())
        print("\n非糖尿病组数据列:", normal_df.columns.tolist())
        print("\n非糖尿病组缺失值统计:")
        print(normal_df.isnull().sum())

        # 生成表1
        print("\n生成表1...")
        table1 = generate_table1(gdm_df, normal_df)

        # 显示表1
        print("\n" + "=" * 80)
        print("Table 1. Clinical characteristics of study population classified by group")
        print("=" * 80)
        print(table1.to_string(index=False))

        # 保存到Excel
        output_path = r"D:\Desktop\lw\Table1_Clinical_Characteristics.xlsx"
        table1.to_excel(output_path, index=False)
        print(f"\n表1已保存到: {output_path}")

    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        print("请确保文件路径正确，并且文件存在。")
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()