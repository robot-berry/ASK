# 六图真实流程小样本测试

比较范围：原题图片 → 本地 OCR（含公式识别）→ 真实模型流式调用 → 工程校验和本地修正 → 六张 PNG 发布。没有标准答案对比，也没有据此统计解题正确率。

四模型使用原工程提示词、JSON Schema、90 秒模型预算和学弱层后 15 秒预算；为控制费用统一限制输出 4096 tokens，关闭提供商修复重试。GLM 启用思考，其余候选关闭思考。逐层时间是后端发布到任务状态的时间，不含网页轮询、传输和浏览器绘制。

样本上限：数学 10 题、物理 10 题，每模型各 20 题。每题每模型一次，属于探索性小样本，不能证明长期等效。若累计费用预留达到上限则停止。基线首题 OCR 冷启动异常单独归档并补测，原费用仍计入账本。

| 模型 | 已测 | 六图合规 | 模型完整五层 | 严格校验通过* | 首层解析均秒* | 六图首次齐全均秒* | 任务完成均秒* | 逐层发布例数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen_max | 18 | 18 | 15 | 11 | 5.211 | 18.215 | 21.34 | 18 |
| kimi | 17 | 17 | 17 | 6 | 40.201 | 40.546 | 40.559 | 0 |
| glm_flash | 17 | 17 | 17 | 6 | 16.485 | 16.874 | 16.89 | 0 |
| qwen_27b | 18 | 18 | 15 | 8 | 44.698 | 45.013 | 45.026 | 0 |

*均值和严格校验统计限模型完整五层且六图合规的样本；不把兜底或部分内容补齐当作完整模型解析。严格校验只检查工程的呈现约束，不等于答案正确性。首次图片可能为暂定版本，任务完成还包含校验、重绘；两种时延分别列示。

## 同题配对比较

| 候选 | 可配对题数 | OCR 文本完全一致题数 | 六图齐全时延差均值（候选减基线，秒） |
|---|---:|---:|---:|
| kimi | 15 | 15 | 18.712 |
| glm_flash | 15 | 14 | -2.113 |
| qwen_27b | 13 | 13 | 23.851 |

OCR 每次重新运行，可能因本地超时或识别差异改变附图策略；不一致样本应结合逐题记录解释，不能全部归因于模型。

## 费用

上轮保守计费 3.732776 元；本轮保守计费/预留 6.032820 元；累计 9.765596 / 10 元。已报告用量的估算合计 5.185367 元。未知用量保留调用前上限预留，已知用量保留 10% 余量；最终扣费以供应商账单为准。待返回调用 0 次。

## 逐题审计

| 题目 | 模型 | 状态 | 六图首次齐全秒 | 完整模型五层 | 严格校验 | 兜底 | 部分骨架 | 结果目录 |
|---|---|---|---:|---|---|---|---|---|
| TZIP-MATH-T01-Q01 | kimi | completed | 33.3261 | True | False | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/math/20261011T235459_ef451a03d70e43c98cb8150e7fa46916) |
| TZIP-MATH-T01-Q01 | glm_flash | completed | 15.904 | True | False | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/math/20261011T235532_140afa7c5f2c4d9daf3b072560b1f151) |
| TZIP-MATH-T01-Q01 | qwen_27b | completed | 27.1782 | True | True | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/math/20261011T235548_6cc18c11d4b24dbc8f9e9a008f494b8d) |
| TZIP-MATH-T01-Q01 | qwen_max | completed | 12.1702 | True | True | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/math/20261011T235805_db33bc33c66c4595a8c9fba0afa2e461) |
| TZIP-MATH-T01-Q02 | kimi | completed | 24.8145 | True | True | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/math/20261011T235820_7d3b24414722424fbd7abac81fcee30d) |
| TZIP-MATH-T01-Q02 | glm_flash | completed | 14.5984 | True | True | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/math/20261011T235845_f8adcd2859d442158fb2ac88d1ab0622) |
| TZIP-MATH-T01-Q02 | qwen_27b | completed | 31.4498 | True | True | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/math/20261011T235859_a75bac774bae43cc82e8661532089165) |
| TZIP-MATH-T01-Q02 | qwen_max | completed | 12.9509 | True | True | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/math/20261011T235931_1d1bceb6a7ec4d30a556ad5297237811) |
| TZIP-MATH-T01-Q03 | glm_flash | completed | 19.3172 | True | False | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/math/20261011T235947_0c0dda57da1c494989eddf1126c510a6) |
| TZIP-MATH-T01-Q03 | qwen_27b | completed | 44.6371 | True | True | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/math/20261012T000006_737d24b920d24b87a77bfdac3aa582fc) |
| TZIP-MATH-T01-Q03 | qwen_max | completed | 17.7117 | True | True | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/math/20261012T000051_cd50e608bd1342db98cc45257b310584) |
| TZIP-MATH-T01-Q03 | kimi | completed | 36.1977 | True | False | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/math/20261012T000111_e4b019f8f97846e1b09e8ee01c090378) |
| TZIP-MATH-T01-Q05 | qwen_27b | completed | 46.7695 | True | True | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/math/20261012T000147_41629d3a0a4640629ac79e2970d4b967) |
| TZIP-MATH-T01-Q05 | qwen_max | completed | 14.9448 | True | True | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/math/20261012T000234_776c6fe4c2ac417d93a2c95e5bc61967) |
| TZIP-MATH-T01-Q05 | kimi | completed | 33.3512 | True | True | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/math/20261012T000251_56e1bcf02f1b4379883c12067acc082b) |
| TZIP-MATH-T01-Q05 | glm_flash | completed | 11.9059 | True | False | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/math/20261012T000325_8731520bafdb43fb9aae69841b3e1475) |
| TZIP-MATH-T01-Q07 | qwen_max | completed | 20.5395 | True | True | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/math/20261012T000337_87e1de7552484e35a6866155f6c88bbb) |
| TZIP-MATH-T01-Q07 | kimi | completed | 38.4798 | True | False | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/math/20261012T000401_8ae1c7435078493fabd1c4f7dfcdf452) |
| TZIP-MATH-T01-Q07 | glm_flash | completed | 15.3734 | True | True | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/math/20261012T000440_5368261a4fd34504852e4c4f31512177) |
| TZIP-MATH-T01-Q07 | qwen_27b | completed | 33.051 | True | True | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/math/20261012T000455_4f602f1e47274b2bbfbc980148686603) |
| TZIP-MATH-T01-Q09 | kimi | completed | 44.8637 | True | False | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/math/20261012T000528_be748a0fdd25438f9c7911cd7e755d89) |
| TZIP-MATH-T01-Q09 | glm_flash | completed | 14.9024 | True | True | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/math/20261012T000613_bbdc891cbb2b4d72b22efff5dcaa2d63) |
| TZIP-MATH-T01-Q09 | qwen_27b | completed | 38.7879 | True | True | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/math/20261012T000628_9c065d982e384d6597be165477cfa1e5) |
| TZIP-MATH-T01-Q09 | qwen_max | completed | 18.8019 | True | True | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/math/20261012T000707_b9f4896aa43149b2adac52913a03dc47) |
| TZIP-MATH-T01-Q10 | glm_flash | completed | 12.699 | True | True | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/math/20261012T000728_afe62d2747c14afd88605da5653f1ffc) |
| TZIP-MATH-T01-Q10 | qwen_27b | completed | 55.9795 | True | True | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/math/20261012T000741_f3460498c9dd4b2ca4dda7f86facda04) |
| TZIP-MATH-T01-Q10 | qwen_max | completed | 11.5735 | True | True | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/math/20261012T000837_17bcfd372f814524b1e4dd49a4950a28) |
| TZIP-MATH-T01-Q10 | kimi | completed | 36.5645 | True | False | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/math/20261012T000851_a3270b9beb0b478a9607581626c0614c) |
| TZIP-MATH-T01-Q12 | qwen_27b | completed | 91.9262 | False | False | True | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/math/20261012T000928_b034f9addd7f4adb8999201ffc2e1d05) |
| TZIP-MATH-T01-Q12 | qwen_max | completed | 24.3799 | True | True | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/math/20261012T001100_29a36db9695a4246905da69f6815bb89) |
| TZIP-MATH-T01-Q12 | kimi | completed | 48.5057 | True | True | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/math/20261012T001128_46e5cdd5cc8f4edbb90d68b96acec04e) |
| TZIP-MATH-T01-Q12 | glm_flash | completed | 16.993 | True | False | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/math/20261012T001217_298559eb38004d16910dc7e335097306) |
| TZIP-MATH-T01-Q13 | qwen_max | completed | 24.7009 | True | True | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/math/20261012T001234_b39db6ad630d49a094ee3a360bb3e127) |
| TZIP-MATH-T01-Q13 | kimi | completed | 47.452 | True | True | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/math/20261012T001302_f57247b2cfc54f8cac8b698b389ab860) |
| TZIP-MATH-T01-Q13 | glm_flash | completed | 19.2999 | True | True | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/math/20261012T001349_4fe1d70bdfb94b1bbe710871760c4a81) |
| TZIP-MATH-T01-Q13 | qwen_27b | completed | 92.1069 | False | False | True | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/math/20261012T001408_9fc3ed636db1429fb218b6a105c2bfcc) |
| TZIP-MATH-T01-Q15 | kimi | completed | 54.5853 | True | True | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/math/20261012T001541_97f698b226ab410d888fbd3455f6b935) |
| TZIP-MATH-T01-Q15 | glm_flash | completed | 21.9331 | True | False | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/math/20261012T001635_e151646cd45d453abb740fe86c64aa67) |
| TZIP-MATH-T01-Q15 | qwen_27b | completed | 87.5083 | True | False | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/math/20261012T001657_fe1fa508a8c042b78a9625a50fd06585) |
| TZIP-MATH-T01-Q15 | qwen_max | completed | 49.8974 | False | False | False | True | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/math/20261012T001825_f9cc77b879cb4ab08fc649ca3f035457) |
| TZIP-PHYS-T01-Q01 | qwen_max | completed | 23.2828 | True | True | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/physics/20261011T235617_ce0c98e174b64624b21c8e926313f9fe) |
| TZIP-PHYS-T01-Q01 | kimi | completed | 20.3995 | True | False | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/physics/20261011T235642_be70bd4c656d4d52b54386995ef4412f) |
| TZIP-PHYS-T01-Q01 | glm_flash | completed | 12.3474 | True | True | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/physics/20261011T235703_edce182bcbcb48158987a6d9707ebd2c) |
| TZIP-PHYS-T01-Q01 | qwen_27b | completed | 24.7561 | True | False | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/physics/20261011T235715_26763a9f43094b4696c2e050d9cba810) |
| TZIP-PHYS-T01-Q03 | kimi | completed | 37.5764 | True | False | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/physics/20261012T001917_d7b368fcb81b404ea521636c6ec8728d) |
| TZIP-PHYS-T01-Q03 | glm_flash | completed | 23.1659 | True | False | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/physics/20261012T001955_75798ee594634d9b8d841b3e45fc0289) |
| TZIP-PHYS-T01-Q03 | qwen_27b | completed | 51.2488 | True | False | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/physics/20261012T002018_bba1af9ae03b4b28bbd30fc6e379d453) |
| TZIP-PHYS-T01-Q03 | qwen_max | completed | 20.4237 | True | False | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/physics/20261012T002109_3f0df63761b04537932f3ca07c34b5d4) |
| TZIP-PHYS-T01-Q05 | glm_flash | completed | 17.1659 | True | False | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/physics/20261012T002134_1b1904e5b33d40a48eb0f549c4e90832) |
| TZIP-PHYS-T01-Q05 | qwen_27b | completed | 43.4915 | True | False | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/physics/20261012T002151_f463eea3e7294cc0afee1d09fd43c720) |
| TZIP-PHYS-T01-Q05 | qwen_max | completed | 15.3952 | True | False | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/physics/20261012T002235_672eacac166743b7b475834995f27cac) |
| TZIP-PHYS-T01-Q05 | kimi | completed | 29.9997 | True | False | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/physics/20261012T002255_0f6dc832589e4ae285f791c1d0a7a9a8) |
| TZIP-PHYS-T01-Q07 | qwen_27b | completed | 50.9086 | True | True | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/physics/20261012T002325_210a88db4b454564bc0c6d76e11d0415) |
| TZIP-PHYS-T01-Q07 | qwen_max | completed | 25.8512 | True | True | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/physics/20261012T002416_ce2fc55ab50f43e4b22caa0211d52563) |
| TZIP-PHYS-T01-Q07 | kimi | completed | 58.1401 | True | True | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/physics/20261012T002444_3293e4a1185649488e868edf2b3047ac) |
| TZIP-PHYS-T01-Q07 | glm_flash | completed | 16.6414 | True | False | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/physics/20261012T002543_47587507965a44b8bfdc8c460b34c630) |
| TZIP-PHYS-T01-Q08 | qwen_max | completed | 15.43 | True | False | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/physics/20261012T002559_d51f6d8d8ffc47ff9b84f1bbb4e81ec6) |
| TZIP-PHYS-T01-Q08 | kimi | completed | 34.7106 | True | False | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/physics/20261012T002619_32ad010e3d8c4198a7b4d536d5eabe7c) |
| TZIP-PHYS-T01-Q08 | glm_flash | completed | 14.6253 | True | False | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/physics/20261012T002653_03c177f14a904f5a8791a5c6708e1c97) |
| TZIP-PHYS-T01-Q08 | qwen_27b | completed | 49.0316 | True | False | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/physics/20261012T002708_432d3eb932704914a65c075280a20d7c) |
| TZIP-PHYS-T01-Q09 | kimi | completed | 29.5237 | True | False | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/physics/20261012T002757_fd2aeab60ca3450786fa2dcd7c2def09) |
| TZIP-PHYS-T01-Q09 | glm_flash | completed | 16.5812 | True | False | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/physics/20261012T002827_5138a00bc7af4055b2a800bbb85f2467) |
| TZIP-PHYS-T01-Q09 | qwen_27b | completed | 36.9125 | True | False | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/physics/20261012T002843_83215c9a80864250acbef4a2fbbe0453) |
| TZIP-PHYS-T01-Q09 | qwen_max | completed | 15.0648 | True | False | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/physics/20261012T002920_b5b61a601dd14f32ab20678eaec7020e) |
| TZIP-PHYS-T01-Q10 | glm_flash | completed | 23.4124 | True | False | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/physics/20261012T002938_98efb67bbb1c4871904b6d0433ed96b5) |
| TZIP-PHYS-T01-Q10 | qwen_27b | completed | 92.2887 | False | False | True | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/physics/20261012T003002_dd05b8381913479fa701e07f8e404e4b) |
| TZIP-PHYS-T01-Q10 | qwen_max | completed | 62.1271 | False | False | False | True | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/physics/20261012T003134_bfabd02667db4504b1c1b90f3da27e44) |
| TZIP-PHYS-T01-Q10 | kimi | completed | 80.7851 | True | False | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/physics/20261012T003237_33c7559706744e9cbf3b55938cca97ec) |
| TZIP-PHYS-T01-Q11 | qwen_27b | completed | 53.4843 | True | False | False | False | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/physics/20261012T003358_c115c6fca5a04781afc247c86a24d703) |
| TZIP-PHYS-T01-Q11 | qwen_max | completed | 44.3297 | False | False | False | True | [六图及审计](D:/robot/Documents/ASK/picture/test6/benchmark/six_image_compare_20261012/jobs/physics/20261012T003451_b6351fb231c6409e9822e7500dfd6484) |

## 逐层发布兼容性

工程的 tier_streaming.py 使用 completed_ordered_prefix，要求根字段从 l1 开始依次闭合。已对保存的 GLM 数学首题响应做离线复现：原始字段顺序 f,l1,l2,l3,l4,l5,pat,s,u 不能触发 L1 检测；只将相同内容重排为工程指定顺序后能够触发。此检查不新增调用、不更改生产逻辑，也不证明实时适配后的时延。

| 模型 | 响应首字段为 l1 | 与基线 OCR 文本及附图数量一致的配对 | 这些配对六图时延差均值秒 |
|---|---:|---:|---:|
| qwen_max | 15 | 15 | 0.0 |
| kimi | 0 | 15 | 18.712 |
| glm_flash | 0 | 14 | -2.379 |
| qwen_27b | 0 | 13 | 23.851 |

## 停止条件与样本覆盖

实际完成付费调用的正式组合为 70/80，另有一次基线冷启动调用归档；1 个任务在调用前被预算保护阻止，不计作模型失败。预算保护停止后没有继续尝试便宜模型填满余额，避免进一步扩大样本不平衡。

原始结果、逐图首次发布时间和全部校验警告见 results_math.json、results_physics.json；调用用量与预算见 ledger.json。

## 结果判断与图片抽查

GLM-5.3-Flash 的六图齐全速度最接近基线，但首层解析更晚出现，严格格式校验通过比例较低，当前证据不支持直接等效替换。Kimi 和 Qwen3.8-27B 在本次工程接入方式下整体更慢。建议继续保留基线；若后续另行授权适配与测试，优先处理 GLM 的字段顺序与公式格式兼容。

基线有 3 例在 L5 发布后触发 post_lower_budget_exhausted，保留了模型的五层内容，但尾部公式/总结未完整接收；为统一统计口径，将其排除在完整响应时延均值之外。它们与 Qwen27B 的 3 例超时通用兜底不同。

已核验 70 组正式结果的 420 个 PNG 均可解码且哈希与记录一致。人工查看 GLM 数学首题六图总览，以及物理 Q10 四个模型的第六图：所查页面文字可读、段落边界清楚，没有明显截断；Qwen27B 的物理 Q10 页面为重复的通用步骤，印证兜底标记。GLM 有些公式以线性文本显示；基线截流例缺少尾部基础公式信息。此处是呈现与内容结构抽查，不是所有页面的人工验收，也未比较标准答案。

六图文件存在和 PNG 尺寸合规不代表题干信息齐全。工程警告中仍包含原题引用图片但 source_figures 为空、公式定界符或下标格式异常等问题，且部分问题也发生在基线。故无法仅靠出图率断言解析效果相同。
