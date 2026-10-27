# -*- coding: utf-8 -*-
"""
合成企业语料生成器（方案 A：规模压测数据）

基于现有示例智能测试数据的结构，程序化生成同风格的虚构企业文档：
- docx: 制度条款 / 劳动合同 / 入职登记表 / 报销单（保留章条结构、表格、表单字段）
- xlsx: 销售台账 / 工资表（保留合并单元格、多 sheet、number_format——压 OpenpyxlParser）
- md/csv: 公司简介 / 供应商名录 / 通讯录

变异维度：虚构公司名、人名、产品名、金额、日期、比例、章节标题——
结构保真、语义全新，专压解析/分块/去重/检索链路，不压语义真实性。

断点续传：progress.jsonl 记录已生成文件，中断重跑自动跳过；
随机种子按 doc_id 固定 → 同参数重跑产出一致，已生成文件哈希稳定幂等。

用法:
    python generate_corpus.py --target 200 --out stress_corpus/synthetic
    python generate_corpus.py --target 500 --out stress_corpus/synthetic  # 增量续跑
"""
import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime

# ============================================================
# 词料池（全部虚构，避免与真实企业/个人重名）
# ============================================================

COMPANIES = [
    "云杉智控", "沧澜精密", "临渊自动化", "青崖机器人", "磐石传感",
    "洛珈电子", "若水智能装备", "岱宗仪器", "望舒科技", "凌虚控制",
    "鸿蒙工业软件", "苍梧测控", "白泽系统", "青鸾智造", "昆吾机电",
    "荆山激光", "沄涛液压", "澈云光电", "琥珀工业", "烛龙仪表",
]

PERSON_SURNAMES = "王李张刘陈杨赵黄周吴徐孙马朱胡郭何高林罗郑梁谢宋唐许韩冯邓曹彭"
PERSON_GIVEN = [
    "思远", "振海", "维安", "宇航", "嘉树", "云帆", "清和", "知行", "沐宸", "修远",
    "安然", "博衍", "敬亭", "慕言", "承宇", "亦琛", "少卿", "书航", "明煦", "砚青",
]

PRODUCTS = [
    ("XY", "工业振动传感器"), ("LN", "在线监测终端"), ("QS", "边缘采集网关"),
    ("HM", "预测性维护主机"), ("BL", "无线测振节点"),
]

CITIES = ["苏州", "宁波", "无锡", "常州", "嘉兴", "佛山", "东莞", "青岛", "郑州", "合肥"]

DEPARTMENTS = ["研发部", "生产部", "质量部", "采购部", "销售部", "财务部", "人事行政部", "信息技术部"]

# ============================================================
# 工具
# ============================================================


def make_rng(doc_key) -> random.Random:
    """按 doc_key 派生独立随机源——同 key 永远产出同内容（幂等）"""
    if isinstance(doc_key, int):
        seed = doc_key
    else:
        seed = int.from_bytes(str(doc_key).encode("utf-8"), "big") % (2**32)
    return random.Random(seed)


def rand_person(rng: random.Random) -> str:
    return rng.choice(PERSON_SURNAMES) + rng.choice(PERSON_GIVEN)


def rand_amount(rng: random.Random, low=1000, high=500000) -> float:
    return round(rng.uniform(low, high), 2)


def rand_date(rng: random.Random, year=2026) -> str:
    return f"{year}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"


def rand_ratio(rng: random.Random, lo=0.02, hi=0.45) -> float:
    return round(rng.uniform(lo, hi), 4)


# ============================================================
# docx 生成（python-docx）
# ============================================================


def gen_policy_docx(path: str, company: str, rng: random.Random):
    """管理制度：章/条结构 + 版本信息 + 责任部门——压条款分块"""
    from docx import Document

    topics = [
        ("信息安全与保密管理", "涉密文件须通过公司加密系统传输，严禁使用个人社交软件传递商业秘密", None),
        ("差旅与费用报销管理", None, "hotel"),
        ("绩效考核管理", None, "ratio"),
        ("采购管理", None, "limit"),
    ]
    topic, key_clause_tpl, key_type = rng.choice(topics)
    version = f"V{rng.randint(1, 3)}.{rng.randint(0, 9)}"
    hotel = rng.choice([300, 400, 500, 600])
    s_ratio = rand_ratio(rng, 0.1, 0.3)
    limit = rng.choice([5000, 10000, 20000])
    if topic == "差旅与费用报销管理":
        key_clause = f"员工出差住宿费凭发票据实报销，单次出差住宿费上限为每日{hotel}元"
    elif topic == "绩效考核管理":
        key_clause = f"年度绩效考核结果分为S/A/B/C/D五级，其中S级比例不超过{s_ratio:.0%}"
    elif topic == "采购管理":
        key_clause = f"单笔采购金额超过{limit}元须进行三家比价，并经分管副总审批"
    else:
        key_clause = "涉密文件须通过公司加密系统传输，严禁使用个人社交软件传递商业秘密"

    doc = Document()
    doc.add_heading(f"{company}{topic}制度", level=0)
    doc.add_paragraph(f"版本：{version}（现行）    发布日期：{rand_date(rng)}    归口部门：{rng.choice(DEPARTMENTS)}")

    for ch in range(1, rng.randint(4, 6)):
        doc.add_heading(f"第{ch}章 " + rng.choice(["总则", "管理职责", "流程规范", "监督与考核", "附则"]), level=1)
        for t in range(1, rng.randint(3, 6)):
            if ch == 2 and t == 2:
                body = key_clause
            else:
                body = (
                    f"{company}各部门应当严格执行本制度相关规定，违反规定的按照公司"
                    f"《员工奖惩条例》第{rng.randint(2, 9)}章处理；相关审批流程自提交之日起"
                    f"{rng.randint(1, 5)}个工作日内完成，特殊情况可延长至{rng.randint(6, 10)}个工作日。"
                )
            doc.add_paragraph(f"第{ch}.{t}条  {body}")
    doc.save(path)

    qas = [{"q": f"{company}{topic}制度的版本号是多少？", "a": version}]
    if key_type == "hotel":
        qas.append({"q": f"{company}差旅制度规定单次出差住宿费上限是每日多少元？", "a": f"{hotel}元"})
    elif key_type == "limit":
        qas.append({"q": f"{company}采购制度规定单笔采购金额超过多少元须进行三家比价？", "a": f"{limit}元"})
    return qas


def gen_contract_docx(path: str, company: str, rng: random.Random):
    """合同：甲乙方 + 标的表 + 条款——压表格与长文本混合"""
    from docx import Document

    person = rand_person(rng)
    partner = rng.choice(COMPANIES) + "有限公司"
    kind = rng.choice(["劳动合同", "保密与竞业限制协议", "技术服务合同", "采购框架合同"])
    warranty = rng.choice([12, 18, 24, 36])
    penalty = rand_ratio(rng, 0.05, 0.3)
    contract_no = f"HT-{rng.randint(2026000, 2026999)}"
    doc = Document()
    doc.add_heading(f"{kind}_{person}", level=0)
    doc.add_paragraph(f"甲方：{company}有限公司")
    doc.add_paragraph(f"乙方：{person if '劳动' in kind or '保密' in kind else partner}")
    doc.add_paragraph(f"签订日期：{rand_date(rng)}    合同编号：{contract_no}")

    doc.add_heading("一、合同标的", level=1)
    table = doc.add_table(rows=1, cols=4)
    table.style = "Table Grid"
    for i, h in enumerate(["序号", "项目名称", "单价（元）", "数量"]):
        table.rows[0].cells[i].text = h
    first_item, first_price = None, None
    for i in range(1, rng.randint(3, 7)):
        code, name = rng.choice(PRODUCTS)
        price = rand_amount(rng, 500, 80000)
        if first_item is None:
            first_item, first_price = f"{code}-{rng.randint(100, 999)} {name}", price
        row = table.add_row()
        row.cells[0].text = str(i)
        row.cells[1].text = f"{code}-{rng.randint(100, 999)} {name}"
        row.cells[2].text = f"{price:,.2f}"
        row.cells[3].text = str(rng.randint(1, 200))

    doc.add_heading("二、权利与义务", level=1)
    for n in range(1, rng.randint(4, 8)):
        doc.add_paragraph(f"{n}.1 乙方应当保证所提供产品/服务符合国家相关标准，"
                          f"质保期为自验收合格之日起{warranty}个月。")
    doc.add_heading("三、违约责任", level=1)
    doc.add_paragraph(f"任何一方违约的，应向守约方支付合同总额{penalty:.0%}的违约金；"
                      f"争议协商不成的，提交{rng.choice(CITIES)}仲裁委员会仲裁。")
    doc.save(path)
    qas = [
        {"q": f"{kind}_{person}的合同编号是多少？", "a": contract_no},
        {"q": f"{kind}_{person}约定的质保期是多少个月？", "a": f"{warranty}个月"},
    ]
    if first_item:
        qas.append({"q": f"{kind}_{person}中标的第一项产品名称是什么？", "a": first_item})
    return qas


def gen_form_docx(path: str, company: str, rng: random.Random):
    """表单类：入职登记表/报销单——压勾选框字段与键值对抽取"""
    from docx import Document

    person = rand_person(rng)
    kind = rng.choice(["新员工入职登记表", "差旅费报销单", "员工请假申请单"])
    doc = Document()
    doc.add_heading(f"{kind}_{person}", level=0)
    doc.add_paragraph(f"填表日期：{rand_date(rng)}    部门：{rng.choice(DEPARTMENTS)}")

    table = doc.add_table(rows=1, cols=2)
    table.style = "Table Grid"
    fields = [
        ("姓名", person), ("性别", rng.choice(["男", "女"])),
        ("学历", edu := rng.choice(["本科", "硕士", "大专", "博士"])),
        ("联系电话", phone := f"1{rng.choice('3589')}{rng.randint(10000000, 99999999)}"),
        ("入职日期" if "入职" in kind else "申请日期", rand_date(rng)),
        ("紧急联系人", rand_person(rng)),
        ("银行卡号", f"6222{rng.randint(10000000000000, 99999999999999)}"),
        ("期望薪资/报销金额（元）", amt := f"{rand_amount(rng, 4000, 30000):,.2f}"),
    ]
    for k, v in fields:
        row = table.add_row()
        row.cells[0].text = k
        row.cells[1].text = v
    doc.add_paragraph(f"审批意见：同意。  审批人：{rand_person(rng)}  {rand_date(rng)}")
    doc.save(path)
    return [
        {"q": f"{kind}中{person}的学历是什么？", "a": edu},
        {"q": f"{kind}中{person}的联系电话是多少？", "a": phone},
        {"q": f"{kind}中{person}填写的金额是多少元？", "a": amt},
    ]


# ============================================================
# xlsx 生成（openpyxl——保留合并单元格与 number_format）
# ============================================================


def gen_ledger_xlsx(path: str, company: str, rng: random.Random):
    """销售台账：真实数字 + number_format 百分比列——压数字渲染与打码误伤"""
    import openpyxl
    from openpyxl.utils import get_column_letter

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "销售订单"
    headers = ["订单编号", "客户名称", "下单日期", "产品型号", "数量", "含税金额（元）", "回款比例", "负责人"]
    ws.append(headers)
    partners = [rng.choice(COMPANIES) + "有限公司" for _ in range(6)]
    orders = []
    for i in range(rng.randint(20, 60)):
        code, name = rng.choice(PRODUCTS)
        no = f"SO-{rng.randint(202600, 202699)}{i:03d}"
        amt = rand_amount(rng, 2000, 900000)
        owner = rand_person(rng)
        cust = rng.choice(partners)
        orders.append((no, amt, owner, cust))
        ws.append([no, cust, rand_date(rng),
                   f"{code}-{rng.randint(100, 999)} {name}",
                   rng.randint(1, 500), amt,
                   rand_ratio(rng, 0.1, 1.0), owner])
    for col, fmt in zip(range(3, 9), [None, None, None, None, "#,##0.00", "0.00%", None]):
        if fmt:
            letter = get_column_letter(col)
            for cell in ws[letter][1:]:
                cell.number_format = fmt
    # 合计行（合并单元格——压合并填充逻辑）
    total_row = ws.max_row + 1
    ws.cell(row=total_row, column=1, value="合计")
    ws.merge_cells(start_row=total_row, start_column=1, end_row=total_row, end_column=4)
    ws.cell(row=total_row, column=6, value=f"=SUM(F2:F{total_row - 1})").number_format = "#,##0.00"

    ws2 = wb.create_sheet("季度汇总")
    ws2.append(["季度", "订单数", "销售额（元）", "环比"])
    q_top = None
    for q in range(1, 5):
        qn, qv = rng.randint(30, 90), rand_amount(rng, 100000, 5000000)
        if q_top is None or qv > q_top[2]:
            q_top = (f"2026Q{q}", qn, qv)
        ws2.append([f"2026Q{q}", qn, qv, rand_ratio(rng, -0.3, 0.4)])
    wb.save(path)
    no, amt, owner, cust = orders[0]
    top = max(orders, key=lambda x: x[1])
    return [
        {"q": f"{company}销售订单台账中订单编号为{no}的负责人是谁？", "a": owner},
        {"q": f"{company}销售订单台账中含税金额最高的订单金额是多少元？", "a": f"{top[1]:,.2f}元（订单{top[0]}，客户{top[3]}）"},
        {"q": f"{company}销售订单台账中2026年哪个季度的销售额最高？", "a": f"{q_top[0]}（{q_top[2]:,.2f}元）"},
    ]


def gen_payroll_xlsx(path: str, company: str, rng: random.Random):
    """工资表：合并单元格标题 + 多 sheet——压 OpenpyxlParser 的合并填充"""
    import openpyxl
    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter

    wb = openpyxl.Workbook()
    wb.remove(wb.active)  # 删除默认空 Sheet
    qa_month, qa_emp, qa_pay, qa_dept = None, None, None, None
    for mi, m in enumerate(rng.sample(range(1, 13), 2)):
        ws = wb.create_sheet(f"{m}月工资")
        ws.merge_cells("A1:H1")
        ws["A1"] = f"{company}有限公司 {m} 月工资表"
        ws["A1"].font = Font(bold=True, size=14)
        ws.append(["工号", "姓名", "部门", "基本工资", "绩效工资", "社保扣款", "个税", "实发工资"])
        emps = []
        for i in range(rng.randint(10, 30)):
            name = rand_person(rng)
            dept = rng.choice(DEPARTMENTS)
            base = round(rng.uniform(5000, 25000), 2)
            perf = round(base * rand_ratio(rng, 0.1, 0.5), 2)
            social = round(base * 0.105, 2)
            tax = round(max(0, (base + perf - social - 5000) * 0.1), 2)
            net = round(base + perf - social - tax, 2)
            emps.append((name, dept, net))
            ws.append([f"GH{rng.randint(1000, 9999)}", name, dept,
                       base, perf, social, tax, net])
        for col in range(4, 9):
            letter = get_column_letter(col)
            for cell in ws[letter][2:]:
                cell.number_format = "#,##0.00"
        if mi == 0 and emps:
            qa_month, qa_emp, qa_pay, qa_dept = m, emps[0][0], emps[0][2], emps[0][1]
    wb.save(path)
    if qa_emp:
        return [{
            "q": f"{company}有限公司 {qa_month} 月工资表中 {qa_emp} 的实发工资是多少元？"
                 f"其在哪个部门？",
            "a": f"{qa_pay:,.2f}元，{qa_dept}",
        }]
    return []


# ============================================================
# 文本类生成（md / csv）
# ============================================================


def gen_profile_md(path: str, company: str, rng: random.Random):
    city = rng.choice(CITIES)
    code, pname = rng.choice(PRODUCTS)
    year = rng.randint(2008, 2020)
    capital = rand_amount(rng, 5000000, 80000000)
    rev = rand_amount(rng, 30000000, 900000000)
    rev_growth = rand_ratio(rng, 0.05, 0.6)
    rd_ratio = rand_ratio(rng, 0.06, 0.2)
    staff = rng.randint(80, 2000)
    lines = [
        f"# {company}有限公司简介",
        "",
        f"{company}有限公司成立于{year}年，注册资本{capital:,.0f}万元，",
        f"总部位于{city}高新区，是一家专注于工业振动监测与预测性维护的高新技术企业。",
        "",
        "## 主营产品",
        f"- {code}-系列 {pname}：覆盖{rng.choice(['加速度', '速度', '位移'])}量程 {rng.randint(1, 50)}g，采样率最高 {rng.randint(10, 102)}kHz",
        f"- 预测性维护云平台：接入设备超过 {rng.randint(1000, 50000)} 台，故障预警准确率 {rand_ratio(rng, 0.85, 0.98):.1%}",
        "",
        "## 经营情况",
        f"{rng.choice([2024, 2025])}年营业收入 {rev:,.0f} 元，同比增长 {rev_growth:.1%}；",
        f"研发投入占比 {rd_ratio:.1%}，员工总数 {staff} 人。",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return [
        {"q": f"{company}有限公司成立于哪一年？", "a": f"{year}年"},
        {"q": f"{company}有限公司的注册资本是多少万元？", "a": f"{capital:,.0f}万元"},
        {"q": f"{company}有限公司总部位于哪个城市？", "a": city},
        {"q": f"{company}有限公司的员工总数是多少人？", "a": f"{staff}人"},
    ]


def gen_csv(path: str, company: str, rng: random.Random):
    if "供应商" in path:
        rows = [["供应商名称", "城市", "合作状态", "年供货额（元）", "联系人"]]
        for _ in range(rng.randint(15, 40)):
            rows.append([rng.choice(COMPANIES) + "有限公司", rng.choice(CITIES),
                         rng.choice(["合作中", "暂停", "已淘汰"]),
                         f"{rand_amount(rng, 50000, 3000000):,.2f}", rand_person(rng)])
    else:
        rows = [["工号", "姓名", "部门", "职位", "手机号"]]
        for _ in range(rng.randint(20, 60)):
            rows.append([f"GH{rng.randint(1000, 9999)}", rand_person(rng), rng.choice(DEPARTMENTS),
                         rng.choice(["专员", "主管", "经理", "总监", "工程师"]),
                         f"1{rng.choice('3589')}{rng.randint(10000000, 99999999)}"])
    with open(path, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)


# ============================================================
# 主流程：每家虚构公司一套文档 + checkpoint
# ============================================================

GENS = [
    ("{cat}/公司简介_{co}.md", gen_profile_md, "06_产品与项目"),
    ("{cat}/{co}销售订单台账.xlsx", gen_ledger_xlsx, "07_数据清单"),
    ("{cat}/{co}工资表.xlsx", gen_payroll_xlsx, "02_财务表格"),
    ("{cat}/{co}{topic}制度.docx", None, "05_制度流程"),      # 动态命名
    ("{cat}/{kind}_{p}.docx", None, "03_合同协议"),
    ("{cat}/{kind2}_{p2}.docx", None, "04_人事行政"),
    ("{cat}/{co}供应商名录.csv", None, "07_数据清单"),
]


def generate_one_company(co_idx: int, out_dir: str) -> tuple:
    company = COMPANIES[co_idx % len(COMPANIES)] + f"{co_idx // len(COMPANIES) + 1}" * (co_idx >= len(COMPANIES))
    cat_root = os.path.join(out_dir, f"虚拟企业{co_idx:03d}_{company}")
    os.makedirs(cat_root, exist_ok=True)
    made, qas = [], []

    def rel(p):
        made.append(p)
        return p

    r = make_rng(co_idx * 10 + 1)
    p = os.path.join(cat_root, f"公司简介_{company}.md")
    for qa in gen_profile_md(rel(p), company, r):
        qas.append({**qa, "evidence": p})

    r = make_rng(co_idx * 10 + 2)
    p = os.path.join(cat_root, f"{company}销售订单台账.xlsx")
    for qa in gen_ledger_xlsx(rel(p), company, r):
        qas.append({**qa, "evidence": p})

    r = make_rng(co_idx * 10 + 3)
    p = os.path.join(cat_root, f"{company}工资表.xlsx")
    for qa in gen_payroll_xlsx(rel(p), company, r):
        qas.append({**qa, "evidence": p})

    # 制度文档：制度主题影响文件名 → 用主题级 rng 保证确定性
    from docx import Document  # noqa: F401  确认依赖
    r = make_rng(co_idx * 10 + 4)
    topic_name = r.choice(["信息安全与保密", "差旅费用报销", "绩效考核", "采购"])
    p = os.path.join(cat_root, f"{company}{topic_name}管理制度.docx")
    for qa in gen_policy_docx(rel(p), company, r):
        qas.append({**qa, "evidence": p})

    r = make_rng(co_idx * 10 + 5)
    p = os.path.join(cat_root, f"合同_{rand_person(make_rng(co_idx * 10 + 5))}.docx")
    for qa in gen_contract_docx(rel(p), company, r):
        qas.append({**qa, "evidence": p})

    r = make_rng(co_idx * 10 + 6)
    p = os.path.join(cat_root, f"表单_{rand_person(make_rng(co_idx * 10 + 6))}.docx")
    for qa in gen_form_docx(rel(p), company, r):
        qas.append({**qa, "evidence": p})

    gen_csv(rel(os.path.join(cat_root, f"{company}供应商名录.csv")), company, make_rng(co_idx * 10 + 7))
    return made, qas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=200, help="目标文档总数")
    ap.add_argument("--out", default="stress_corpus/synthetic")
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    ck_path = os.path.join(out, "progress.jsonl")

    # ---- 断点续传：读取已完成记录 ----
    done = set()
    if os.path.exists(ck_path):
        with open(ck_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["path"])
        print(f"[checkpoint] 发现 {len(done)} 个已生成文件，续跑跳过")

    per_co = 7
    need_co = -(-args.target // per_co)
    generated, qa_total, t0 = 0, 0, datetime.now()
    for co_idx in range(need_co):
        made, qas = generate_one_company(co_idx, out)
        # checkpoint 每文件一条 + QA 真值随行；已存在文件跳过（幂等）
        for p in made:
            if p in done:
                continue
            doc_qas = [q for q in qas if q.get("evidence") == p]
            with open(ck_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"path": os.path.abspath(p),
                                    "ts": datetime.now().isoformat(timespec="seconds"),
                                    "qa": doc_qas},
                                   ensure_ascii=False) + "\n")
            generated += 1
            qa_total += len(doc_qas)

    # ---- 汇总：文档 + qa.jsonl ----
    total, by_ext = 0, {}
    all_qas = []
    for root, _, files in os.walk(out):
        for fn in files:
            if fn == "progress.jsonl":
                continue
            total += 1
            ext = os.path.splitext(fn)[1]
            by_ext[ext] = by_ext.get(ext, 0) + 1
    with open(ck_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_qas.extend(json.loads(line).get("qa", []))
    qa_path = os.path.join(out, "qa.jsonl")
    with open(qa_path, "w", encoding="utf-8") as f:
        for qa in all_qas:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")
    print(f"本轮新生成 {generated} 份；目录总计 {total} 份 {by_ext}")
    print(f"QA 标签：{len(all_qas)} 条 → {qa_path}")
    print(f"耗时 {(datetime.now() - t0).total_seconds():.1f}s → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
