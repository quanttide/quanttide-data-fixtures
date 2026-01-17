"""
Questionnaire Inspector

检查 Processor 生成的清洗后数据是否符合 Plan 定义。
"""

import pandas as pd
import re
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path


class QuestionnaireInspector:
    """
    问卷数据检查器

    职责：
    1. 验证清洗后数据是否符合 plan 的数据模型定义
    2. 检查数据质量规则（完整性、范围、一致性）
    3. 生成检查报告
    """

    def __init__(self, plan_path: Path):
        """
        初始化检查器

        Args:
            plan_path: Plan 文件路径
        """
        self.plan_path = plan_path
        self.plan_content = plan_path.read_text(encoding='utf-8')
        self.field_definitions = self._parse_field_definitions()
        self.check_results = []

    def _parse_field_definitions(self) -> List[Dict[str, Any]]:
        """解析 plan 中的字段定义"""
        fields = []
        lines = self.plan_content.split('\n')
        in_table = False
        headers = []
        data_model_start = self.plan_content.find('## 数据模型')

        for i, line in enumerate(lines):
            # 检查是否在数据模型章节内
            if '## 数据模型' in line:
                continue
            # 检查是否遇到其他章节
            if line.startswith('##') and data_model_start != -1:
                line_start = sum(len(l) + 1 for l in lines[:i])
                if line_start > data_model_start and '数据模型' not in line:
                    break

            if line.strip().startswith('|'):
                cells = [c.strip() for c in line.split('|')[1:-1]]

                if '字段名' in cells:
                    headers = [h.strip() for h in cells]
                    in_table = True
                elif in_table and len(cells) == len(headers):
                    # 检查是否是分隔线
                    if any(c.startswith('---') or c == '--------' for c in cells):
                        continue

                    row = dict(zip(headers, cells))
                    # 提取字段名（可能在反引号中）
                    field_name_match = re.search(r'`([^`]+)`', cells[0])
                    if field_name_match:
                        field_name = field_name_match.group(1).strip()
                    else:
                        field_name = cells[0].strip()

                    # 跳过空行
                    if not field_name:
                        continue

                    # 获取各个列的值
                    source_value = row.get('原始来源', '') if '原始来源' in headers else ''
                    type_value = row.get('类型', '') if '类型' in headers else ''
                    desc_value = row.get('值标签/格式规范', '') if '值标签/格式规范' in headers else ''
                    missing_value = row.get('缺失编码', '') if '缺失编码' in headers else ''
                    constraint_value = row.get('逻辑约束', '') if '逻辑约束' in headers else ''

                    # 统一类型名称
                    type_normalized = type_value.strip() if type_value else ''
                    if 'binary' in type_normalized:
                        type_normalized = 'binary'
                    elif 'text' in type_normalized:
                        type_normalized = 'text'
                    elif 'categorical' in type_normalized:
                        type_normalized = 'categorical'

                    fields.append({
                        'name': field_name,
                        'source': source_value.strip() if source_value else '',
                        'type': type_normalized,
                        'description': desc_value.strip() if desc_value else '',
                        'missing_code': self._parse_missing_code(missing_value.strip() if missing_value else ''),
                        'constraints': self._parse_constraints(constraint_value.strip() if constraint_value else '')
                    })

        return fields

    def _parse_missing_code(self, code_str: str) -> Optional[int]:
        """解析缺失编码"""
        # 去除反引号
        code_str = code_str.replace('`', '').strip()

        if not code_str or code_str == '—':
            return None
        if code_str == '-99':
            return -99
        if code_str == '-88':
            return -88
        try:
            return int(code_str)
        except ValueError:
            return None

    def _parse_constraints(self, constraint_str: str) -> Dict[str, Any]:
        """解析约束条件"""
        constraints = {}
        if '必填' in constraint_str:
            constraints['required'] = True
        if '≥' in constraint_str:
            match = re.search(r'≥(\d+)', constraint_str)
            if match:
                constraints['min'] = int(match.group(1))
        if '≤' in constraint_str:
            match = re.search(r'≤(\d+)', constraint_str)
            if match:
                constraints['max'] = int(match.group(1))
        if '–' in constraint_str or (re.search(r'(\d+)[–-](\d+)', constraint_str) and '≤' not in constraint_str):
            match = re.search(r'(\d+)[–-](\d+)', constraint_str)
            if match:
                constraints['min'] = int(match.group(1))
                constraints['max'] = int(match.group(2))
        return constraints

    def validate_schema_compliance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """检查 Schema 合规性（公开方法）"""
        return self._check_schema_compliance(df)

    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """检查数据质量（公开方法）"""
        return self._check_data_quality(df)

    def validate_business_rules(self, df: pd.DataFrame) -> Dict[str, Any]:
        """检查业务规则（公开方法）"""
        return self._check_business_rules(df)

    def inspect(self, cleaned_data: pd.DataFrame) -> Dict[str, Any]:
        """
        执行完整检查

        Args:
            cleaned_data: 清洗后的 DataFrame

        Returns:
            检查结果字典
        """
        results = {
            'schema_compliance': self._check_schema_compliance(cleaned_data),
            'data_quality': self._check_data_quality(cleaned_data),
            'business_rules': self._check_business_rules(cleaned_data),
            'summary': {}
        }

        # 计算汇总信息
        total_issues = (
            len(results['schema_compliance']['issues']) +
            len(results['data_quality']['issues']) +
            len(results['business_rules']['issues'])
        )
        results['summary'] = {
            'total_checks': len(self.field_definitions) + len(results['data_quality']['checks']) + len(results['business_rules']['checks']),
            'passed': total_issues == 0,
            'total_issues': total_issues
        }

        return results

    def _check_schema_compliance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """检查 Schema 合规性"""
        issues = []
        expected_columns = set(field['name'] for field in self.field_definitions)
        actual_columns = set(df.columns)

        # 检查缺失字段
        missing_fields = expected_columns - actual_columns
        if missing_fields:
            issues.append(f"缺失字段: {missing_fields}")

        # 检查多余字段
        extra_fields = actual_columns - expected_columns
        if extra_fields:
            issues.append(f"多余字段: {extra_fields}")

        # 检查字段类型
        for field in self.field_definitions:
            if field['name'] in df.columns:
                dtype_str = str(df[field['name']].dtype)
                expected_type = field['type']

                if expected_type == 'integer':
                    if 'int' not in dtype_str and df[field['name']].notna().any():
                        issues.append(f"字段 {field['name']} 类型应为 integer，实际为 {dtype_str}")
                elif expected_type == 'float':
                    if 'float' not in dtype_str and 'int' not in dtype_str:
                        issues.append(f"字段 {field['name']} 类型应为 float，实际为 {dtype_str}")

        return {
            'status': 'PASS' if not issues else 'FAIL',
            'issues': issues
        }

    def _check_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """检查数据质量"""
        issues = []
        checks = []

        # 检查必填字段
        for field in self.field_definitions:
            if field.get('constraints', {}).get('required'):
                if field['name'] in df.columns:
                    null_count = df[field['name']].isna().sum()
                    if null_count > 0:
                        issues.append(f"必填字段 {field['name']} 有 {null_count} 个空值")
                checks.append(f"必填字段 {field['name']}")

        # 检查数值范围
        for field in self.field_definitions:
            if field['name'] in df.columns and 'min' in field.get('constraints', {}):
                min_val = field['constraints']['min']
                max_val = field['constraints'].get('max')

                if max_val is None:
                    out_of_range = df[df[field['name']] < min_val]
                else:
                    out_of_range = df[(df[field['name']] < min_val) | (df[field['name']] > max_val)]

                missing_code = field.get('missing_code')
                if missing_code is not None and not out_of_range.empty:
                    out_of_range = out_of_range[out_of_range[field['name']] != missing_code]

                if len(out_of_range) > 0:
                    max_str = str(max_val) if max_val else '∞'
                    issues.append(f"字段 {field['name']} 有 {len(out_of_range)} 个值超出范围 [{min_val}, {max_str}]")

                checks.append(f"字段 {field['name']} 范围检查")

        # 检查缺失值编码
        for field in self.field_definitions:
            if field['name'] in df.columns and field.get('missing_code'):
                missing_code = field['missing_code']
                if df[field['name']].dtype in ['int64', 'float64']:
                    valid_codes = [missing_code]
                    invalid_codes = df[df[field['name']].isin(valid_codes)].index
                    # 这是正常情况，不报错
                    checks.append(f"字段 {field['name']} 缺失编码检查")

        return {
            'status': 'PASS' if not issues else 'FAIL',
            'issues': issues,
            'checks': checks
        }

    def _check_business_rules(self, df: pd.DataFrame) -> Dict[str, Any]:
        """检查业务规则"""
        issues = []
        checks = []

        # 规则1：部门与说明一致性
        if 'department' in df.columns and 'other_dept_specify' in df.columns:
            invalid_rows = df[
                (df['department'] != 5) &
                (df['other_dept_specify'].notna()) &
                (df['other_dept_specify'] != '') &
                (df['other_dept_specify'] != '-99')
            ]
            if len(invalid_rows) > 0:
                issues.append(f"有 {len(invalid_rows)} 行：非其他部门但填写了说明")
            checks.append("部门与说明一致性")

        # 规则2：满意度与工作负荷范围
        if 'satisfaction' in df.columns:
            # 获取缺失编码
            satisfaction_field = next((f for f in self.field_definitions if f['name'] == 'satisfaction'), None)
            missing_code = satisfaction_field.get('missing_code') if satisfaction_field else None

            # 先排除缺失值编码和 NaN
            mask = df['satisfaction'].notna()
            if missing_code is not None:
                mask = mask & (df['satisfaction'] != missing_code)

            # 检查范围
            valid_satisfaction = df[mask]
            invalid_satisfaction = valid_satisfaction[
                (valid_satisfaction['satisfaction'] < 1) | (valid_satisfaction['satisfaction'] > 5)
            ]

            if len(invalid_satisfaction) > 0:
                issues.append(f"满意度有 {len(invalid_satisfaction)} 个超出 1-5 范围的值")
            checks.append("满意度范围检查")

        if 'workload' in df.columns:
            # 获取缺失编码
            workload_field = next((f for f in self.field_definitions if f['name'] == 'workload'), None)
            missing_code = workload_field.get('missing_code') if workload_field else None

            # 先排除缺失值编码和 NaN
            mask = df['workload'].notna()
            if missing_code is not None:
                mask = mask & (df['workload'] != missing_code)

            # 检查范围
            valid_workload = df[mask]
            invalid_workload = valid_workload[
                (valid_workload['workload'] < 1) | (valid_workload['workload'] > 5)
            ]

            if len(invalid_workload) > 0:
                issues.append(f"工作负荷有 {len(invalid_workload)} 个超出 1-5 范围的值")
            checks.append("工作负荷范围检查")

        # 规则3：福利虚拟变量互斥性（如果有 benefit_insurance 等字段）
        benefit_cols = [col for col in df.columns if col.startswith('benefit_') and col != 'benefits_raw']
        for col in benefit_cols:
            if col in df.columns:
                invalid_values = df[~df[col].isin([0, 1])]
                if len(invalid_values) > 0:
                    issues.append(f"字段 {col} 有 {len(invalid_values)} 个非 0/1 的值")
                checks.append(f"{col} 值检查")

        return {
            'status': 'PASS' if not issues else 'FAIL',
            'issues': issues,
            'checks': checks
        }

    def generate_report(self, inspection_result: Dict[str, Any]) -> str:
        """生成检查报告"""
        report = []
        report.append("=" * 60)
        report.append("问卷数据检查报告")
        report.append("=" * 60)
        report.append("")

        # 汇总信息
        summary = inspection_result['summary']
        status_icon = "✅" if summary['passed'] else "❌"
        report.append(f"总体状态: {status_icon} {'通过' if summary['passed'] else '未通过'}")
        report.append(f"总检查项: {summary['total_checks']}")
        report.append(f"问题数量: {summary['total_issues']}")
        report.append("")

        # Schema 合规性
        report.append("-" * 60)
        report.append("Schema 合规性")
        report.append("-" * 60)
        schema_result = inspection_result['schema_compliance']
        report.append(f"状态: {schema_result['status']}")
        if schema_result['issues']:
            for issue in schema_result['issues']:
                report.append(f"  ❌ {issue}")
        else:
            report.append("  ✅ 所有字段定义正确")
        report.append("")

        # 数据质量
        report.append("-" * 60)
        report.append("数据质量")
        report.append("-" * 60)
        quality_result = inspection_result['data_quality']
        report.append(f"状态: {quality_result['status']}")
        if quality_result['issues']:
            for issue in quality_result['issues']:
                report.append(f"  ❌ {issue}")
        else:
            report.append("  ✅ 所有数据质量检查通过")
        report.append("")

        # 业务规则
        report.append("-" * 60)
        report.append("业务规则")
        report.append("-" * 60)
        business_result = inspection_result['business_rules']
        report.append(f"状态: {business_result['status']}")
        if business_result['issues']:
            for issue in business_result['issues']:
                report.append(f"  ❌ {issue}")
        else:
            report.append("  ✅ 所有业务规则检查通过")
        report.append("")

        return "\n".join(report)


def main():
    """主函数：示例用法"""
    from pathlib import Path

    # 路径配置
    fixtures_root = Path(__file__).parent.parent
    plan_path = fixtures_root / "plan" / "questionnaire_cleaning_plan.md"
    cleaned_data_path = fixtures_root / "record" / "questionnaire_cleaned.csv"

    # 创建检查器
    inspector = QuestionnaireInspector(plan_path)

    # 读取清洗后数据
    cleaned_data = pd.read_csv(cleaned_data_path)

    # 执行检查
    results = inspector.inspect(cleaned_data)

    # 生成报告
    report = inspector.generate_report(results)
    print(report)

    # 返回检查结果
    return results['summary']['passed']


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
