import ffmpeg
import math


def split_video_ffmpeg(video_file, segment_duration=3):
    # 获取视频的总时长
    probe = ffmpeg.probe(video_file)
    video_duration = float(next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')['duration'])

    # 计算需要切分成多少段
    n_segments = math.ceil(video_duration / segment_duration)

    for i in range(n_segments):
        start_time = i * segment_duration
        # 使用ffmpeg裁剪视频
        (
            ffmpeg
            .input(video_file, ss=start_time, t=segment_duration)
            .output(f"04_01_{i}_r.mp4", c='copy')
            .run(capture_stdout=True, capture_stderr=True)
        )


# 调用函数，参数为视频文件路径
split_video_ffmpeg("D:/DL/drive_data/actor10/12-1.mp4")

