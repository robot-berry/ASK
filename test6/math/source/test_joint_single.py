from pathlib import Path
import json

from search_machine_pipeline import SearchMachinePipeline


# ============================================================
# 1. 要测试的图片
# ============================================================

IMAGE_PATH = Path(
    r"D:\robot\Documents\ASK\P_DATA\test_photo\photo_0004_4.jpg"
)


# ============================================================
# 2. 测试结果根目录
# ============================================================

SOURCE_DIR = Path(__file__).resolve().parent

TEST_ROOT = SOURCE_DIR / "joint_test_output"


def main():

    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(
            f"找不到测试图片：{IMAGE_PATH}"
        )

    TEST_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 创建正式 pipeline
    # --------------------------------------------------------

    pipeline = SearchMachinePipeline(
        runs_root=TEST_ROOT
    )

    print("=" * 70)
    print("开始联合测试")
    print("=" * 70)

    print(f"原图：{IMAGE_PATH}")
    print()

    # --------------------------------------------------------
    # 创建任务
    # --------------------------------------------------------

    state = pipeline.new_job(
        IMAGE_PATH.read_bytes(),
        IMAGE_PATH.name,
    )

    job_id = str(state["job_id"])

    print(f"Job ID：{job_id}")

    job_dir = pipeline.job_path(job_id)

    print(f"任务目录：{job_dir}")
    print()

    # --------------------------------------------------------
    # 运行完整正式流程
    # --------------------------------------------------------

    print("开始运行：")
    print("processor → model → renderer → image validation")
    print()

    state = pipeline.solve(
        job_id,
        use_model=True,
    )

    # --------------------------------------------------------
    # 打印最终状态
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("运行结束")
    print("=" * 70)

    print(
        "状态：",
        state.get("status"),
    )

    # --------------------------------------------------------
    # 预处理结果
    # --------------------------------------------------------

    preprocessed = (
        job_dir
        / "audit"
        / "preprocessed_source.png"
    )

    print()
    print("预处理图：")

    if preprocessed.is_file():
        print(preprocessed)
    else:
        print("未生成")

    # --------------------------------------------------------
    # 模型分析辅助图
    # --------------------------------------------------------

    analysis_reference = (
        job_dir
        / "audit"
        / "printed_text_reference.png"
    )

    print()
    print("文字分析参考图：")

    if analysis_reference.is_file():
        print(analysis_reference)
    else:
        print("未生成（这不一定是错误）")

    # --------------------------------------------------------
    # 最终六张图：一张原题图 + 五档解析图
    # --------------------------------------------------------

    final_names = [
        "01_question_knowledge.png",
        "02_guidance_elite.png",
        "03_guidance_upper.png",
        "04_guidance_middle.png",
        "05_guidance_lower.png",
        "06_guidance_novice.png",
    ]

    print()
    print("最终 Renderer 图片：")

    final_count = 0

    for name in final_names:

        path = job_dir / name

        if path.is_file():
            print(f"[OK] {path}")
            final_count += 1
        else:
            print(f"[--] {name} 未生成")

    print()
    print(
        f"最终图片数量：{final_count}/4"
    )

    # --------------------------------------------------------
    # 预处理审计
    # --------------------------------------------------------

    preprocess = state.get(
        "preprocess"
    )

    if isinstance(preprocess, dict):

        print()
        print("预处理信息：")

        print(
            "rotation =",
            preprocess.get("rotation")
        )

        print(
            "rotation_method =",
            preprocess.get(
                "rotation_method"
            )
        )

        print(
            "deskew_status =",
            preprocess.get(
                "deskew_status"
            )
        )

        print(
            "deskew_applied_angle =",
            preprocess.get(
                "deskew_applied_angle"
            )
        )

        print(
            "perspective_status =",
            preprocess.get(
                "perspective_status"
            )
        )

        print(
            "cleanup_method =",
            preprocess.get(
                "cleanup_method"
            )
        )

    # --------------------------------------------------------
    # 如果失败，把真正错误直接显示
    # --------------------------------------------------------

    error = state.get("error")

    if error:

        print()
        print("=" * 70)
        print("失败原因")
        print("=" * 70)

        print(
            json.dumps(
                error,
                ensure_ascii=False,
                indent=2,
            )
        )

    # --------------------------------------------------------
    # job.json
    # --------------------------------------------------------

    print()
    print("完整 Job 信息：")

    print(
        job_dir / "job.json"
    )


if __name__ == "__main__":
    main()
