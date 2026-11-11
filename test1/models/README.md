# PaddleOCR models

The preprocessing pipeline uses three Apache-2.0 ONNX models:

- `PP-LCNet_x1_0_doc_ori.onnx`: four-way document orientation classifier converted from PaddleOCR `PP-LCNet_x1_0_doc_ori`; SHA-256 `10453f80f6f50e8149161aef61b265119471e4010263a82682e5695914236f27`.
- `PP-LCNet_x1_0_textline_ori.onnx`: official PaddleOCR 0/180-degree text-line classifier; SHA-256 `38aa97cd4be591e0ad304e659f07ba30d946f27a63315433f6659c69c8778345`.
- `PP-OCRv5_mobile_det.onnx`: official lightweight PP-OCRv5 text detector used to locate printed question blocks; SHA-256 `a431985659dc921974177a95adcfbb90fd9e51989a5e04d70d0b75f597b6e61d`.

Sources:

- https://huggingface.co/ningpp/PP-LCNet_x1_0_doc_ori-ONNX
- https://huggingface.co/PaddlePaddle/PP-LCNet_x1_0_textline_ori_onnx
- https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_det_onnx
