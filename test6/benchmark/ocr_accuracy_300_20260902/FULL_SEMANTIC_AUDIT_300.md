# 300题全量OCR原题语义完整性视觉审核

> 范围：仅审核OCR原题文字与保留配图能否完整恢复题意；未调用答题模型；测试集没有标准答案文本，因此本报告不是字符准确率报告。

## 总体结果

| 范围 | 完整 | 不完整 | 完整率 |
|---|---:|---:|---:|
| 全部 | 96 | 204 | 32.00% |
| 数学 | 34 | 116 | 22.67% |
| 物理 | 62 | 88 | 41.33% |

## 当前修正后的语义兜底

- 完整原题像素证据覆盖：**300/300（100.00%）**。
- OCR文字继续用于可复制、可重排的正文；公式、上下标、漏行或选项错位以完整原题像素校验区为准。
- 这项100%是“不丢原题证据”，不是把纯文字OCR准确率改写成100%。

## 判定标准

- 完整：OCR文字与保留配图合起来，题干、已知量、符号、公式、全部选项/分问足以恢复原题语义。
- 不完整：关键题干、数值、单位、指数/上下标、分式/根式、选项或分问有漏失、错换，可能改变解题条件或答案含义。
- 不因少量标点、换行、题号变化判错；手写批注只有在遮蔽或污染原题含义时才判为不完整。

## 高频问题

| 错误类型 | 题数 |
|---|---:|
| `missing_fraction` | 59 |
| `missing_formula` | 54 |
| `formula_corruption` | 41 |
| `subscript_loss` | 39 |
| `exponent_corruption` | 21 |
| `missing_value` | 18 |
| `symbol_substitution` | 14 |
| `missing_options` | 13 |
| `missing_option` | 13 |
| `missing_stem` | 9 |
| `missing_values` | 5 |
| `table_omission` | 4 |
| `missing_subquestion` | 4 |
| `nuclide_notation_corruption` | 4 |
| `missing_subquestions` | 4 |
| `formula_omission` | 3 |
| `missed_stem_and_formula` | 2 |
| `missing_symbol` | 2 |
| `figure_dependent` | 2 |
| `annotation_contamination` | 2 |
| `missing_conditions` | 2 |
| `label_corruption` | 2 |
| `missing_constant` | 2 |
| `nuclear_symbol_corruption` | 2 |
| `handwriting_contamination` | 2 |
| `missing_fraction_denominators` | 1 |
| `inequality_corruption` | 1 |
| `missing_probability` | 1 |
| `missing_exponent` | 1 |
| `missing_minus` | 1 |
| `infinity_corruption` | 1 |
| `missing_conclusion` | 1 |
| `missing_domain` | 1 |
| `missing_question` | 1 |
| `variable_substitution` | 1 |
| `adjacent_figure_contamination` | 1 |
| `missing_option_value` | 1 |
| `symbol_addition` | 1 |
| `missing_reference_formula` | 1 |
| `sign_corruption` | 1 |
| `missing_option_values` | 1 |
| `missing_label` | 1 |
| `missing_blank` | 1 |
| `missing_inequality` | 1 |
| `missing_mass_ratio` | 1 |
| `missing_subquestion_structure` | 1 |
| `extra_question_text` | 1 |
| `fraction_corruption` | 1 |
| `missing_qualifier` | 1 |
| `missing_option_text` | 1 |
| `symbol_corruption` | 1 |

## 分试卷结果

| 学科 | 试卷 | 完整 | 不完整 | 完整率 |
|---|---|---:|---:|---:|
| 数学 | Test1 | 6 | 9 | 40.00% |
| 数学 | Test10 | 8 | 7 | 53.33% |
| 数学 | Test2 | 3 | 12 | 20.00% |
| 数学 | Test3 | 2 | 13 | 13.33% |
| 数学 | Test4 | 4 | 11 | 26.67% |
| 数学 | Test5 | 4 | 11 | 26.67% |
| 数学 | Test6 | 1 | 14 | 6.67% |
| 数学 | Test7 | 1 | 14 | 6.67% |
| 数学 | Test8 | 3 | 12 | 20.00% |
| 数学 | Test9 | 2 | 13 | 13.33% |
| 物理 | Test1 | 6 | 9 | 40.00% |
| 物理 | Test10 | 5 | 10 | 33.33% |
| 物理 | Test2 | 5 | 10 | 33.33% |
| 物理 | Test3 | 7 | 8 | 46.67% |
| 物理 | Test4 | 9 | 6 | 60.00% |
| 物理 | Test5 | 6 | 9 | 40.00% |
| 物理 | Test6 | 5 | 10 | 33.33% |
| 物理 | Test7 | 5 | 10 | 33.33% |
| 物理 | Test8 | 5 | 10 | 33.33% |
| 物理 | Test9 | 9 | 6 | 60.00% |

## 被判定不完整的题目（204题）

| 题号 | 学科/试卷 | 错误类型 | 判定说明 |
|---|---|---|---|
| TZIP-MATH-T01-Q01 | 数学/Test1 | `missed_stem_and_formula` | 整段题干与集合公式漏失；仅保留四个选项 |
| TZIP-MATH-T01-Q03 | 数学/Test1 | `missing_fraction`, `missing_value` | 二项分布参数、概率及分式选项关键数值漏失 |
| TZIP-MATH-T01-Q04 | 数学/Test1 | `subscript_loss`, `symbol_substitution` | 棱锥名称与V1、V2、V3下标损坏导致选项含义不可靠 |
| TZIP-MATH-T01-Q06 | 数学/Test1 | `formula_omission`, `subscript_loss` | 递推式分式结构和数列下标大面积漏失 |
| TZIP-MATH-T01-Q10 | 数学/Test1 | `missing_fraction`, `formula_omission` | l/R与r/R两个关键比值没有保留 |
| TZIP-MATH-T01-Q12 | 数学/Test1 | `missing_fraction`, `missing_value` | 主公式分母、b=2及B=π/4等关键条件漏失 |
| TZIP-MATH-T01-Q13 | 数学/Test1 | `missing_formula`, `missing_fraction` | 双曲线标准式、渐近线系数和距离数值漏失 |
| TZIP-MATH-T01-Q14 | 数学/Test1 | `formula_corruption`, `missing_fraction` | 第(2)问数列通项分式与2的n次幂损坏 |
| TZIP-MATH-T01-Q15 | 数学/Test1 | `missing_fraction`, `symbol_substitution` | 第(3)问MN=1/4 PB被错误转写 |
| TZIP-MATH-T02-Q02 | 数学/Test2 | `missing_fraction`, `missing_options` | 四个角度选项的分母均漏失 |
| TZIP-MATH-T02-Q04 | 数学/Test2 | `exponent_corruption`, `symbol_substitution` | 二进制展开中的下标和指数多处损坏 |
| TZIP-MATH-T02-Q05 | 数学/Test2 | `missing_formula`, `missing_fraction` | 椭圆方程和三角形面积3/2漏失 |
| TZIP-MATH-T02-Q06 | 数学/Test2 | `missing_fraction`, `missing_options` | 四个期望值选项的分母全部漏失 |
| TZIP-MATH-T02-Q08 | 数学/Test2 | `missing_formula`, `missing_option` | 多个π分式漏失且D选项未输出 |
| TZIP-MATH-T02-Q09 | 数学/Test2 | `missing_formula` | D选项求和式与三角函数分式结构损坏 |
| TZIP-MATH-T02-Q10 | 数学/Test2 | `missing_fraction`, `missing_value` | 点P到E的固定距离缺少分母3 |
| TZIP-MATH-T02-Q11 | 数学/Test2 | `missing_fraction`, `formula_corruption` | acosC=b+1/2c被转写为b+c |
| TZIP-MATH-T02-Q12 | 数学/Test2 | `table_omission`, `formula_corruption` | 六年销售表全部数值漏失且回归公式分母损坏 |
| TZIP-MATH-T02-Q13 | 数学/Test2 | `subscript_loss`, `symbol_substitution` | A1与A等关键点下标混淆并混入水印 |
| TZIP-MATH-T02-Q14 | 数学/Test2 | `missing_formula`, `missing_value` | 双曲线标准式及MN长度均漏失 |
| TZIP-MATH-T02-Q15 | 数学/Test2 | `missing_formula`, `missing_fraction` | 三个小问中的分式函数、求和式和不等式均大面积漏失 |
| TZIP-MATH-T03-Q01 | 数学/Test3 | `missing_formula`, `missing_options`, `subscript_loss` | 复数下标与z2/z1分式损坏，多个选项数值漏失 |
| TZIP-MATH-T03-Q02 | 数学/Test3 | `missing_formula` | 1/a与1/\|b\|两个关键分式没有被完整识别 |
| TZIP-MATH-T03-Q04 | 数学/Test3 | `missing_fraction_denominators` | 四个根式选项的分母4均漏失 |
| TZIP-MATH-T03-Q05 | 数学/Test3 | `missing_formula`, `missing_options` | 目标分式函数及含分数的选项关键数值漏失 |
| TZIP-MATH-T03-Q06 | 数学/Test3 | `missing_formula`, `missing_options` | 正切函数系数、f(0)条件及多个选项漏失 |
| TZIP-MATH-T03-Q07 | 数学/Test3 | `missing_formula`, `missing_options` | 双曲线标准式、直线斜率与部分选项被破坏 |
| TZIP-MATH-T03-Q08 | 数学/Test3 | `formula_corruption` | 函数自变量1/x多处漏失，D项指数被错误识别 |
| TZIP-MATH-T03-Q09 | 数学/Test3 | `missing_formula` | cos2α/(1+sin2α)=1/2关键条件未保留 |
| TZIP-MATH-T03-Q10 | 数学/Test3 | `formula_corruption` | 数列递推式与求和目标严重错乱 |
| TZIP-MATH-T03-Q11 | 数学/Test3 | `table_omission`, `formula_corruption` | 主统计表数值全部漏失，列联表与公式不完整 |
| TZIP-MATH-T03-Q13 | 数学/Test3 | `missing_value`, `subscript_loss` | 外接圆半径条件漏失，三条切线下标混淆 |
| TZIP-MATH-T03-Q14 | 数学/Test3 | `formula_corruption`, `inequality_corruption` | 函数系数漏失且第(3)问不等号方向和右端表达式错误 |
| TZIP-MATH-T03-Q15 | 数学/Test3 | `missing_formula`, `missing_options` | 目标分式函数及含分数的选项关键数值漏失 |
| TZIP-MATH-T04-Q02 | 数学/Test4 | `missing_formula`, `missing_options` | 复数模的设问对象漏失，三个含分式的选项损坏 |
| TZIP-MATH-T04-Q04 | 数学/Test4 | `formula_corruption`, `missing_fraction`, `missing_symbol` | 三角函数表达式、φ范围和f(π/3)均严重损坏 |
| TZIP-MATH-T04-Q05 | 数学/Test4 | `formula_corruption`, `missing_fraction` | 指数衰减式损坏且关键分式漏失 |
| TZIP-MATH-T04-Q06 | 数学/Test4 | `missing_formula` | 半角正切条件及sinA/(sinB-sinC)目标式漏失 |
| TZIP-MATH-T04-Q07 | 数学/Test4 | `subscript_loss`, `symbol_substitution` | 四棱台顶点下标与垂直关系中的关键点名损坏 |
| TZIP-MATH-T04-Q08 | 数学/Test4 | `missing_option` | D选项的最小面积数值16√3/9未输出 |
| TZIP-MATH-T04-Q09 | 数学/Test4 | `formula_corruption`, `missing_formula` | 多层对数条件及yz/x求值目标均被破坏 |
| TZIP-MATH-T04-Q12 | 数学/Test4 | `missing_probability`, `table_omission`, `formula_corruption` | 三个概率值、参考临界值表及卡方公式分子漏失 |
| TZIP-MATH-T04-Q13 | 数学/Test4 | `missing_fraction` | 第(2)问B=π/3关键条件漏失 |
| TZIP-MATH-T04-Q14 | 数学/Test4 | `missing_formula`, `missing_subquestion` | b_n定义与第(i)问T4漏失，末项不等式下标损坏 |
| TZIP-MATH-T04-Q15 | 数学/Test4 | `missing_formula` | 函数g(x)=e^x/x-1/x-ln x主体表达式漏失 |
| TZIP-MATH-T05-Q02 | 数学/Test5 | `symbol_substitution` | 补集符号与空集符号被错误替换，集合运算含义不可靠 |
| TZIP-MATH-T05-Q03 | 数学/Test5 | `missing_formula` | 双曲线方程中的x²/4项完全漏失 |
| TZIP-MATH-T05-Q04 | 数学/Test5 | `missing_exponent` | 二项式(2+x)的7次幂指数漏失 |
| TZIP-MATH-T05-Q05 | 数学/Test5 | `missing_formula`, `missing_option` | 三个数的指数/分式定义损坏且C选项漏失 |
| TZIP-MATH-T05-Q07 | 数学/Test5 | `symbol_substitution` | 直线属于平面的符号多处变为空框，空间关系含义不完整 |
| TZIP-MATH-T05-Q08 | 数学/Test5 | `missing_fraction` | 题干及B、C、D选项中的π分式分母全部漏失 |
| TZIP-MATH-T05-Q09 | 数学/Test5 | `missing_fraction` | D项cos(knπ/2)的分母2漏失 |
| TZIP-MATH-T05-Q10 | 数学/Test5 | `missing_fraction` | 圆台轴截面角π/3被识别为π |
| TZIP-MATH-T05-Q11 | 数学/Test5 | `missing_fraction`, `subscript_loss` | 单次概率1/2被识别为1，T_n等关键下标有漏失 |
| TZIP-MATH-T05-Q14 | 数学/Test5 | `missing_formula`, `missing_fraction` | 椭圆标准式、离心率及第(2)问证明分式被破坏 |
| TZIP-MATH-T05-Q15 | 数学/Test5 | `missing_formula`, `subscript_loss` | 函数定义整行漏失，三个极值点下标亦混淆 |
| TZIP-MATH-T06-Q01 | 数学/Test6 | `missing_option`, `symbol_substitution` | 补集符号错误且A选项空集未输出 |
| TZIP-MATH-T06-Q02 | 数学/Test6 | `missing_formula`, `missing_options` | 复数分式及四个分数选项全部漏失 |
| TZIP-MATH-T06-Q03 | 数学/Test6 | `figure_dependent`, `subscript_loss`, `missing_option` | 散点图依赖图像且相关系数下标与选项关系式严重损坏 |
| TZIP-MATH-T06-Q04 | 数学/Test6 | `missing_fraction` | C项π/6与D项π/2的分母漏失 |
| TZIP-MATH-T06-Q06 | 数学/Test6 | `missing_formula`, `subscript_loss` | 双曲线标准式、焦点下标及线段关系式严重损坏 |
| TZIP-MATH-T06-Q07 | 数学/Test6 | `missing_minus`, `infinity_corruption` | B选项负号漏失且无穷符号多处损坏 |
| TZIP-MATH-T06-Q08 | 数学/Test6 | `missing_formula`, `symbol_substitution` | 函数(x-1/2)ln(1-1/x)整体漏失且空集符号错误 |
| TZIP-MATH-T06-Q09 | 数学/Test6 | `missing_formula`, `subscript_loss` | 数列通项公式主体及多个S下标漏失 |
| TZIP-MATH-T06-Q10 | 数学/Test6 | `missing_formula` | 函数中的a/x项漏失 |
| TZIP-MATH-T06-Q11 | 数学/Test6 | `formula_omission`, `annotation_contamination` | 题干文字有所恢复但数列关键公式仍漏失并混入手写内容 |
| TZIP-MATH-T06-Q12 | 数学/Test6 | `missing_fraction` | ∠ABC=π/3与∠APD=π/2均被识别成π |
| TZIP-MATH-T06-Q13 | 数学/Test6 | `subscript_loss` | 第(2)问f(x0)被转写为f(x)，改变待证对象 |
| TZIP-MATH-T06-Q14 | 数学/Test6 | `missing_formula`, `missing_value`, `subscript_loss` | 椭圆方程、给定点坐标、离心率及多个点下标损坏 |
| TZIP-MATH-T06-Q15 | 数学/Test6 | `missing_fraction`, `missing_conclusion` | 最终需证明的p<5/11结论漏失 |
| TZIP-MATH-T07-Q01 | 数学/Test7 | `symbol_substitution`, `missing_symbol` | 交集、包含关系等集合符号损坏 |
| TZIP-MATH-T07-Q02 | 数学/Test7 | `subscript_loss`, `exponent_corruption` | 三个方差符号及四个排序选项的下标、平方均混乱 |
| TZIP-MATH-T07-Q04 | 数学/Test7 | `missing_option` | D选项数值8漏失 |
| TZIP-MATH-T07-Q05 | 数学/Test7 | `missing_formula`, `subscript_loss` | x0下标丢失且C项1/x函数整体漏失 |
| TZIP-MATH-T07-Q06 | 数学/Test7 | `subscript_loss`, `formula_corruption` | 乘积数列定义、T6条件及各选项下标损坏 |
| TZIP-MATH-T07-Q07 | 数学/Test7 | `missing_fraction` | D项对称中心(1,1/2)缺少第二坐标 |
| TZIP-MATH-T07-Q08 | 数学/Test7 | `missing_formula`, `missing_options` | 关键函数式和部分选项仍漏失 |
| TZIP-MATH-T07-Q09 | 数学/Test7 | `missing_formula` | 最终求值目标r1/r2漏失 |
| TZIP-MATH-T07-Q10 | 数学/Test7 | `missing_formula`, `missing_domain`, `missing_question` | 三角函数自变量x/2、定义域和设问均漏失 |
| TZIP-MATH-T07-Q11 | 数学/Test7 | `missing_fraction`, `symbol_substitution` | AE=1/4 AC被转写成AE=-AC |
| TZIP-MATH-T07-Q12 | 数学/Test7 | `missing_formula` | 函数f(x)=e^{-x}+ax-a³定义整行漏失 |
| TZIP-MATH-T07-Q13 | 数学/Test7 | `missing_formula`, `missing_value`, `missing_conditions` | 主等式右端、面积值及三个备选条件全部漏失 |
| TZIP-MATH-T07-Q14 | 数学/Test7 | `missing_formula`, `missing_fraction` | 第(i)问\|DF\|/\|FE\|与第(ii)问面积比3/8漏失 |
| TZIP-MATH-T07-Q15 | 数学/Test7 | `missing_fraction`, `subscript_loss`, `formula_corruption` | 概率1/2、P(2)数值及币种数量下标关系多处损坏 |
| TZIP-MATH-T08-Q02 | 数学/Test8 | `variable_substitution`, `adjacent_figure_contamination` | 水面符号l被识别为数字1且混入第4题图标题 |
| TZIP-MATH-T08-Q03 | 数学/Test8 | `missing_fraction`, `missing_option`, `formula_corruption` | A、B选项分母漏失，C选项整体漏失，D项指数写法损坏 |
| TZIP-MATH-T08-Q04 | 数学/Test8 | `missing_fraction` | 两端(1+sinA)/cosA与(1+sinB)/cosB分式结构漏失 |
| TZIP-MATH-T08-Q05 | 数学/Test8 | `missing_formula`, `missing_fraction`, `subscript_loss` | 双曲线方程、直角条件、焦点下标及B项分式损坏 |
| TZIP-MATH-T08-Q06 | 数学/Test8 | `exponent_corruption` | 主函数e^x及选项中的e²均退化为线性字符，指数含义不可靠 |
| TZIP-MATH-T08-Q07 | 数学/Test8 | `symbol_substitution` | χ²独立性检验被转写为x²独立性检验 |
| TZIP-MATH-T08-Q08 | 数学/Test8 | `missing_option`, `subscript_loss` | B选项整体漏失且定义中的S_i、S_j下标丢失 |
| TZIP-MATH-T08-Q09 | 数学/Test8 | `missed_stem_and_formula` | 只识别到填空题章节标题 |
| TZIP-MATH-T08-Q11 | 数学/Test8 | `missing_fraction` | 第(2)问目标AE/BE分式漏失 |
| TZIP-MATH-T08-Q12 | 数学/Test8 | `missing_fraction`, `subscript_loss` | ∠ABC=π/3漏失，B1点名和折叠后图形下标多处丢失 |
| TZIP-MATH-T08-Q13 | 数学/Test8 | `formula_corruption`, `subscript_loss`, `missing_fraction` | 递推式n³、S_{n-1}及b_n=S_n/n均被破坏 |
| TZIP-MATH-T08-Q15 | 数学/Test8 | `exponent_corruption`, `symbol_substitution` | 曲线y=a^x及多个a^m目标的指数被错误转写 |
| TZIP-MATH-T09-Q01 | 数学/Test9 | `exponent_corruption`, `missing_option_value` | 四个科学计数法选项的10次幂指数均有漏失或损坏 |
| TZIP-MATH-T09-Q02 | 数学/Test9 | `missing_options`, `figure_dependent` | 主视图及A-D四个图形选项没有转成可检索文字 |
| TZIP-MATH-T09-Q03 | 数学/Test9 | `subscript_loss` | 两组数据的x1至xn下标关系被破坏 |
| TZIP-MATH-T09-Q04 | 数学/Test9 | `symbol_addition` | “过点B作平行线”被转写为“过点B1作”，引入错误点名 |
| TZIP-MATH-T09-Q05 | 数学/Test9 | `missing_reference_formula`, `sign_corruption` | 参考恒等式漏失且B、D根式中的负号被转成正号 |
| TZIP-MATH-T09-Q06 | 数学/Test9 | `missing_formula` | 关于x的不等式组与关于y的分式方程主体大面积漏失 |
| TZIP-MATH-T09-Q07 | 数学/Test9 | `missing_conditions`, `missing_option_values` | OD:DE、直角与D坐标条件及多个选项数值漏失 |
| TZIP-MATH-T09-Q09 | 数学/Test9 | `missing_fraction` | C项∠DEC=3θ/2的表达式漏失 |
| TZIP-MATH-T09-Q10 | 数学/Test9 | `table_omission` | 幻方题干已恢复但表格内三个已知数未转成文字 |
| TZIP-MATH-T09-Q11 | 数学/Test9 | `missing_fraction`, `annotation_contamination` | 第④组概率1/5漏失，并把手写答案混入第(1)问 |
| TZIP-MATH-T09-Q12 | 数学/Test9 | `missing_formula`, `formula_corruption` | φ=(√5-1)/2漏失，最终AB/AC=φ被转写为AC=φ |
| TZIP-MATH-T09-Q14 | 数学/Test9 | `missing_fraction` | 培育费用8x-4800/x中的4800/x漏失 |
| TZIP-MATH-T09-Q15 | 数学/Test9 | `missing_stem`, `missing_values` | 材料阅读开头的直角三角形条件、AB=6与BC=4漏失 |
| TZIP-MATH-T10-Q04 | 数学/Test10 | `missing_value` | 结论③中的b²=9/4漏失9/4 |
| TZIP-MATH-T10-Q08 | 数学/Test10 | `missing_values`, `formula_corruption` | AB=8m、AD=6m及1/3触发条件等关键信息损坏 |
| TZIP-MATH-T10-Q09 | 数学/Test10 | `missing_fraction`, `missing_label` | 菱形ABCD名称缺字且sinB=4/5漏失 |
| TZIP-MATH-T10-Q10 | 数学/Test10 | `exponent_corruption` | (1/2)^(-1)中的-1指数被错误识别 |
| TZIP-MATH-T10-Q12 | 数学/Test10 | `missing_formula` | 一次函数mx-2m+4与反比例函数k/x均严重漏失 |
| TZIP-MATH-T10-Q14 | 数学/Test10 | `missing_subquestion` | 第(2)问②“弦DF长度的最小值”整体漏失 |
| TZIP-MATH-T10-Q15 | 数学/Test10 | `missing_blank`, `formula_corruption`, `subscript_loss` | c的填空漏失，一次函数公式和交点坐标下标严重损坏 |
| TZIP-PHYS-T01-Q02 | 物理/Test1 | `missing_option`, `nuclide_notation_corruption` | A选项整体漏失，核素质量数/原子序数及反应式记号多处损坏 |
| TZIP-PHYS-T01-Q04 | 物理/Test1 | `missing_option`, `missing_formula` | A选项整体漏失，B选项的关键关系式也未识别 |
| TZIP-PHYS-T01-Q05 | 物理/Test1 | `subscript_loss` | 初速度v₀在题干和多个选项中被识别为v□/vo，关键物理量记号不稳定 |
| TZIP-PHYS-T01-Q06 | 物理/Test1 | `formula_corruption`, `subscript_loss` | L₁/L₂、m₁/m₂等下标损坏，B选项感应电荷公式严重缺项 |
| TZIP-PHYS-T01-Q09 | 物理/Test1 | `missing_fraction` | B选项的3/4T被识别为T，关键时间条件漏失 |
| TZIP-PHYS-T01-Q12 | 物理/Test1 | `missing_stem`, `missing_values`, `label_corruption` | 开头圆弧轨道及R、L、M等关键条件整体漏失，轨道标签也有误 |
| TZIP-PHYS-T01-Q13 | 物理/Test1 | `formula_corruption` | 关键定义式K=B²L²/(2R)的分式结构被破坏 |
| TZIP-PHYS-T01-Q14 | 物理/Test1 | `missing_fraction`, `missing_value` | 磁感应强度系数2√3/3和MN=L/2等关键数值关系漏失 |
| TZIP-PHYS-T01-Q15 | 物理/Test1 | `missing_subquestions` | OCR仅保留题干首段，后续全部求解小问整体漏失 |
| TZIP-PHYS-T02-Q02 | 物理/Test2 | `label_corruption` | 点电荷q_A被识别为q4，关键电荷标签发生歧义 |
| TZIP-PHYS-T02-Q06 | 物理/Test2 | `subscript_loss`, `formula_corruption` | v₁、v₂及k₁等关键下标损坏，线性关系式记号不完整 |
| TZIP-PHYS-T02-Q07 | 物理/Test2 | `missing_option` | 干涉题只识别A、B、C三个选项，D选项漏失 |
| TZIP-PHYS-T02-Q08 | 物理/Test2 | `missing_formula` | B选项最大电流及C选项电荷量的两个关键分式整体漏失 |
| TZIP-PHYS-T02-Q09 | 物理/Test2 | `missing_formula` | 第(3)问给定的a₂=g/8关系式未被完整识别 |
| TZIP-PHYS-T02-Q10 | 物理/Test2 | `missing_subquestion`, `subscript_loss` | 第(3)问Q=385J等前置条件整体漏失，初态量下标亦损坏 |
| TZIP-PHYS-T02-Q11 | 物理/Test2 | `subscript_loss` | m₀、B₁、B₂等贯穿三问的关键下标被识别为方框或普通数字 |
| TZIP-PHYS-T02-Q12 | 物理/Test2 | `formula_corruption`, `missing_fraction` | 巴耳末公式、电势能式和角动量量子化式等核心公式结构严重漏失 |
| TZIP-PHYS-T02-Q13 | 物理/Test2 | `missing_value`, `missing_formula`, `missing_constant` | m3与摩擦因数漏失且弹性势能式缺少系数 |
| TZIP-PHYS-T02-Q15 | 物理/Test2 | `missing_formula` | 四个答案选项中的根式与分式数值几乎全部漏失，只剩选项字母 |
| TZIP-PHYS-T03-Q02 | 物理/Test3 | `formula_corruption`, `exponent_corruption` | 引力波功率公式的分式、c⁵及G⁴/M^m/R^n指数结构严重损坏 |
| TZIP-PHYS-T03-Q06 | 物理/Test3 | `missing_formula` | C、D选项的两个核心分式均只剩分子，分母整体漏失 |
| TZIP-PHYS-T03-Q07 | 物理/Test3 | `formula_corruption`, `missing_fraction`, `exponent_corruption` | 逃逸速度、光子能量/质量关系及常数科学计数法多处公式结构损坏 |
| TZIP-PHYS-T03-Q08 | 物理/Test3 | `missing_fraction`, `formula_corruption` | 给定势能式及C选项均漏失1/2系数，关键物理表达式不完整 |
| TZIP-PHYS-T03-Q10 | 物理/Test3 | `missing_formula` | 劲度系数k=16mg/(5d)整体漏失，C选项5/36系数也漏失 |
| TZIP-PHYS-T03-Q13 | 物理/Test3 | `formula_corruption` | 弹簧劲度系数k=p₀S/H和活塞质量m=p₀S/(10g)两式严重损坏 |
| TZIP-PHYS-T03-Q14 | 物理/Test3 | `missing_value`, `missing_constant` | 圆弧半径与重力常量漏失 |
| TZIP-PHYS-T03-Q15 | 物理/Test3 | `missing_stem`, `missing_subquestions` | OCR只提取了图中坐标刻度，整道题干和三个求解小问均漏失 |
| TZIP-PHYS-T04-Q06 | 物理/Test4 | `formula_corruption`, `missing_fraction` | 引力势能式和总机械能式的分母漏失，根式比值也有损坏 |
| TZIP-PHYS-T04-Q09 | 物理/Test4 | `subscript_loss` | s₁=2.40cm的关键测量编号被识别为s□，与s₂至s₄的序列记号不一致 |
| TZIP-PHYS-T04-Q11 | 物理/Test4 | `missing_subquestion` | 第(2)问“状态1到状态2的内能变化”整体漏失 |
| TZIP-PHYS-T04-Q12 | 物理/Test4 | `missing_fraction`, `formula_corruption` | t₂=t₁+π/(2ω)及ω=(M+m)π/(mt₁)等关键关系式结构漏失 |
| TZIP-PHYS-T04-Q13 | 物理/Test4 | `missing_fraction`, `subscript_loss` | L₁/L₂等长度下标损坏，弹性势能式漏失1/2系数 |
| TZIP-PHYS-T04-Q14 | 物理/Test4 | `nuclide_notation_corruption`, `exponent_corruption` | 氘氚聚变方程的核素上下标严重损坏，科学计数法指数符号亦不稳定 |
| TZIP-PHYS-T05-Q01 | 物理/Test5 | `exponent_corruption` | 四个选项的4.6×10¹⁷均被识别为4.6×1017，指数结构丢失 |
| TZIP-PHYS-T05-Q05 | 物理/Test5 | `subscript_loss` | 0～t₁被识别为0～t，关键时间节点下标漏失 |
| TZIP-PHYS-T05-Q06 | 物理/Test5 | `missing_values`, `subscript_loss` | 参数表中的额定电压220V和小风扇110V漏失，n₁/n₂线圈编号也损坏 |
| TZIP-PHYS-T05-Q08 | 物理/Test5 | `missing_fraction`, `formula_corruption` | A选项3/2系数与C选项π/(3ω)分式结构漏失，公式选项失真 |
| TZIP-PHYS-T05-Q11 | 物理/Test5 | `formula_corruption`, `subscript_loss` | 径向场强E_r=kU/r²被截成k，D选项U'=m₁U/m₂也严重损坏 |
| TZIP-PHYS-T05-Q12 | 物理/Test5 | `missing_subquestions` | OCR只保留装置题干，后续三个实验/计算小问整体漏失 |
| TZIP-PHYS-T05-Q13 | 物理/Test5 | `missing_fraction`, `missing_value` | 水平圆管半径R₂=3/(2π)m整体漏失 |
| TZIP-PHYS-T05-Q14 | 物理/Test5 | `missing_subquestions`, `exponent_corruption` | 两个求解小问整体漏失，普朗克常量10⁻³⁴还被识别为10⁻⁴ |
| TZIP-PHYS-T05-Q15 | 物理/Test5 | `exponent_corruption` | 重复题中的四个4.6×10¹⁷同样均丢失指数结构 |
| TZIP-PHYS-T06-Q02 | 物理/Test6 | `exponent_corruption` | 国际单位制选项中的负指数和上标损坏 |
| TZIP-PHYS-T06-Q03 | 物理/Test6 | `missing_formula` | C、D选项的时间与水平距离分式整体漏失 |
| TZIP-PHYS-T06-Q05 | 物理/Test6 | `missing_fraction`, `formula_corruption` | B选项U_AC=2E_p/q漏失分母q，C选项电场强度式整体漏失 |
| TZIP-PHYS-T06-Q07 | 物理/Test6 | `exponent_corruption` | 四个长度选项中的10⁻⁶、10⁻⁵均被识别为10-6、10-5，指数结构丢失 |
| TZIP-PHYS-T06-Q08 | 物理/Test6 | `missing_fraction`, `formula_corruption` | A至D四个选项中的周期、安培力、焦耳热和电荷量分式普遍漏失 |
| TZIP-PHYS-T06-Q09 | 物理/Test6 | `nuclide_notation_corruption`, `exponent_corruption` | 氘氚氦核素上下标损坏，10²³、10⁻¹⁹、10¹⁴、10⁻¹⁵等指数均丢失 |
| TZIP-PHYS-T06-Q10 | 物理/Test6 | `missing_options`, `missing_fraction` | A、B选项整体漏失，D选项7/2T的分数系数也漏失 |
| TZIP-PHYS-T06-Q12 | 物理/Test6 | `exponent_corruption`, `subscript_loss` | 10⁻⁴m²和10⁵Pa的指数结构丢失，p₁、T₁、T₂等下标被扁平化 |
| TZIP-PHYS-T06-Q13 | 物理/Test6 | `missing_fraction`, `subscript_loss` | 霍尔电压式U_H=kIB/d漏失分母d，D₂等关键侧面编号损坏 |
| TZIP-PHYS-T06-Q14 | 物理/Test6 | `missing_values`, `missing_fraction` | 第(3)、(4)问给定的两个20/3m/s²加速度数值均漏失 |
| TZIP-PHYS-T07-Q01 | 物理/Test7 | `missing_formula` | 单位组合中的根号分式√(N·m/(Ω·s))整体漏失，只剩N·m |
| TZIP-PHYS-T07-Q02 | 物理/Test7 | `missing_formula` | 重复题同样漏失单位组合的完整根号分式 |
| TZIP-PHYS-T07-Q05 | 物理/Test7 | `missing_fraction`, `missing_value` | 高度比例及D选项关键数值漏失 |
| TZIP-PHYS-T07-Q06 | 物理/Test7 | `subscript_loss` | λ₁/λ₂及P₁/P₂、S₁/S₂等贯穿设问的光学下标多处损坏 |
| TZIP-PHYS-T07-Q07 | 物理/Test7 | `missing_fraction`, `formula_corruption` | B选项√(md/(qU))的根号分式结构漏失 |
| TZIP-PHYS-T07-Q08 | 物理/Test7 | `missing_stem`, `nuclide_notation_corruption`, `exponent_corruption` | 中子衰变的定义和核素说明前半段漏失，反应式上下标及科学计数法指数损坏 |
| TZIP-PHYS-T07-Q09 | 物理/Test7 | `missing_formula` | B选项kU/(k²+2)只剩分母，C选项功率分式整体漏失 |
| TZIP-PHYS-T07-Q11 | 物理/Test7 | `missing_stem`, `missing_option` | 两列波的传播方向及波源位置等开头条件漏失，D选项整体漏失 |
| TZIP-PHYS-T07-Q12 | 物理/Test7 | `formula_corruption`, `missing_fraction` | 霍尔电压和片内合电场的C、D两个公式被截成零散分子 |
| TZIP-PHYS-T07-Q15 | 物理/Test7 | `exponent_corruption` | p₀=1.0×10⁵Pa被识别为1.0×105Pa，关键压强指数结构丢失 |
| TZIP-PHYS-T08-Q01 | 物理/Test8 | `missing_inequality`, `subscript_loss` | 第三问关键条件r₁<D/2被识别成“r1已知”，不等关系丢失 |
| TZIP-PHYS-T08-Q02 | 物理/Test8 | `missing_mass_ratio`, `missing_subquestion_structure` | 长题干已恢复但质量比和部分小问结构仍漏失 |
| TZIP-PHYS-T08-Q04 | 物理/Test8 | `nuclear_symbol_corruption`, `extra_question_text` | 核反应上下标损坏并混入上一题文字 |
| TZIP-PHYS-T08-Q05 | 物理/Test8 | `missing_options`, `formula_corruption` | 卫星题四个选项的核心公式大面积漏失，只剩选项字母和零散指数 |
| TZIP-PHYS-T08-Q08 | 物理/Test8 | `missing_fraction`, `formula_corruption` | C选项位移公式分母k漏失，关键公式不完整 |
| TZIP-PHYS-T08-Q09 | 物理/Test8 | `missing_formula` | 导轨双棒题四个选项的公式值均未识别，只保留文字框架 |
| TZIP-PHYS-T08-Q11 | 物理/Test8 | `formula_corruption`, `missing_fraction` | 入射速度eBR/m和B选项周期公式的除法结构丢失 |
| TZIP-PHYS-T08-Q13 | 物理/Test8 | `missing_formula` | 变压器实验B项U₁/U₂与n₁/n₂的关系式整体漏失 |
| TZIP-PHYS-T08-Q14 | 物理/Test8 | `exponent_corruption` | ΔU=2×10⁴J被识别为2×104J，关键数量级指数结构丢失 |
| TZIP-PHYS-T08-Q15 | 物理/Test8 | `fraction_corruption` | 第三问位置条件b/4被识别成b/2，关键距离条件错误 |
| TZIP-PHYS-T09-Q02 | 物理/Test9 | `missing_stem`, `formula_corruption` | 质量、电梯加速度等题干条件整体漏失，C、D选项表达式也被手写内容污染 |
| TZIP-PHYS-T09-Q04 | 物理/Test9 | `formula_corruption`, `handwriting_contamination` | B选项100√3N被识别为1003N100N，C项末尾混入手写内容 |
| TZIP-PHYS-T09-Q05 | 物理/Test9 | `handwriting_contamination`, `symbol_substitution` | 手写演算混入且v0与g等符号被替换 |
| TZIP-PHYS-T09-Q08 | 物理/Test9 | `missing_option` | 速度时间图题A选项主体漏失，只留下无法还原含义的零散文字 |
| TZIP-PHYS-T09-Q09 | 物理/Test9 | `missing_qualifier` | B选项“夹角越来越小”的关键“小”字漏失，选项语义不完整 |
| TZIP-PHYS-T09-Q12 | 物理/Test9 | `missing_stem` | “探究弹力和弹簧伸长关系”的实验题总述漏失，输出直接从图轴和分问开始 |
| TZIP-PHYS-T10-Q01 | 物理/Test10 | `nuclear_symbol_corruption`, `exponent_corruption` | ¹⁴C、¹⁴N等核素上下标全部丢失，10¹⁵也被拉平成1015 |
| TZIP-PHYS-T10-Q04 | 物理/Test10 | `missing_fraction`, `formula_corruption` | B、C、D三个选项中的根式、分式和幂次大面积漏失，无法还原选项数值 |
| TZIP-PHYS-T10-Q05 | 物理/Test10 | `subscript_loss`, `missing_option_text` | t₁至t₄的时刻下标被拉平，C项开头及D项第二个时刻无法区分 |
| TZIP-PHYS-T10-Q06 | 物理/Test10 | `missing_stem`, `missing_value` | 题干开头的斜面倾角、斜面质量等关键条件漏失，输出从木板长度中段开始 |
| TZIP-PHYS-T10-Q07 | 物理/Test10 | `missing_stem`, `missing_value` | 波沿x轴传播、t=1s波形和P点位置等总题干漏失 |
| TZIP-PHYS-T10-Q08 | 物理/Test10 | `missing_value`, `missing_formula` | 质子电荷量q漏失，C项位置表达式和D项场强比公式严重残缺 |
| TZIP-PHYS-T10-Q10 | 物理/Test10 | `missing_fraction`, `missing_formula` | 两段电场做功的分式系数损坏，B、C时间式分母漏失，D项末值整体漏失 |
| TZIP-PHYS-T10-Q11 | 物理/Test10 | `missing_value`, `exponent_corruption` | 舱体积单位幂次、初温及大气压10⁵数量级等关键量漏失或损坏 |
| TZIP-PHYS-T10-Q12 | 物理/Test10 | `missing_value`, `formula_corruption` | 初速度与摩擦因数等关键量漏失或损坏 |
| TZIP-PHYS-T10-Q13 | 物理/Test10 | `subscript_loss`, `symbol_corruption` | F₂在题干和四个选项中持续被识别为空框，两个安培力无法可靠区分 |
