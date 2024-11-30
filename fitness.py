import librosa
import sounddevice as sd
import numpy as np

def extract_middle_segment(y, sr, duration=10):
    # 전체 길이 계산
    total_length = len(y)
    total_duration = total_length / sr
    
    # 중앙의 시작 시간 계산
    middle_start = (total_duration / 2) - (duration / 2)
    
    # 시작 및 끝 인덱스 계산
    start_index = int(middle_start * sr)
    end_index = start_index + int(duration * sr)

    # 잘라낸 부분 반환
    return y[start_index:end_index]

def extract_start_segment(y, sr, duration=10):
    # 시작 부분에서 duration만큼 잘라내기
    return y[:int(duration * sr)]

def fitness_time_domain(file1, file2, duration=10):
    # 오디오 파일 불러오기
    y1, sr1 = librosa.load(file1, sr=None)
    y2, sr2 = librosa.load(file2, sr=None)
    
    # 샘플링 레이트가 서로 다를 경우를 대비해 리샘플링
    if sr1 != sr2:
        y2 = librosa.resample(y2, orig_sr=sr2, target_sr=sr1)

    # 중앙의 일부 구간을 잘라냄
    y1_segment = extract_start_segment(y1, sr1, duration)
    y2_segment = extract_start_segment(y2, sr1, duration)
    
    # 두 신호의 길이를 맞추기 위해 최소 길이로 자르기
    """
    min_len = min(len(y1), len(y2))
    y1 = y1[:min_len]
    y2 = y2[:min_len]

    # 진폭 차이를 통해 유사도 계산
    difference = np.abs(y1 - y2)
    similarity_score = 1 - np.mean(difference)
    
    return similarity_score, y1, y2, sr1
    """
    min_len = min(len(y1_segment), len(y2_segment))
    y1_segment = y1_segment[:min_len]
    y2_segment = y2_segment[:min_len]

    # 진폭 차이를 통해 유사도 계산
    difference = np.abs(y1_segment - y2_segment)
    similarity_score = 1 - np.mean(difference)
    
    return similarity_score, y1_segment, y2_segment, sr1

def fitness_frequency_domain(file1, file2, duration=10, tolerance=1.0):
    # 오디오 파일 불러오기
    y1, sr1 = librosa.load(file1, sr=None)
    y2, sr2 = librosa.load(file2, sr=None)
    
    # 샘플링 레이트가 서로 다를 경우를 대비해 리샘플링
    if sr1 != sr2:
        y2 = librosa.resample(y2, orig_sr=sr2, target_sr=sr1)

    y1_segment = extract_start_segment(y1, sr1, duration)
    y2_segment = extract_start_segment(y2, sr1, duration)

    # 두 스펙트로그램의 길이를 맞추기 위해 최소 길이로 자르기
    min_len = min(len(y1_segment), len(y2_segment))
    y1_segment = y1_segment[:min_len]
    y2_segment = y2_segment[:min_len]

    # STFT를 사용하여 스펙트로그램 생성
    S1 = np.abs(librosa.stft(y1_segment))
    S2 = np.abs(librosa.stft(y2_segment))

    """
    # STFT를 사용하여 스펙트로그램 생성
    S1 = np.abs(librosa.stft(y1))
    S2 = np.abs(librosa.stft(y2))
    """

    # 두 스펙트로그램의 길이를 맞추기 위해 최소 길이로 자르기
    min_frames = min(S1.shape[1], S2.shape[1])
    S1 = S1[:, :min_frames]
    S2 = S2[:, :min_frames]

    # 각 프레임에 대해 벡터 차이 계산
    difference = np.linalg.norm(S1 - S2, axis=0)
    
    # 평균 차이를 통해 유사도 계산
    mean_difference = np.mean(difference)
    
    # 유사도는 1에서 평균 차이를 나누어 반영
    # 차이가 크면 유사도가 낮아지고, 차이가 작으면 유사도가 높아짐
    similarity_score = 1 / (1 + mean_difference / tolerance)

    #return similarity_score, y1, y2, sr1
    return similarity_score, y1_segment, y2_segment, sr1

def play_audio(y, sr):
    # sounddevice 라이브러리를 이용해 오디오 재생
    sd.play(y, sr)
    sd.wait()

if __name__ == "__main__":
    file1 = "./datasets/audio_dance_4.wav"
    file2 = "./datasets/audio_dance_2.wav"
    similarity_score, y1, y2, sr = fitness_time_domain(file1, file2)
    print(f"Time Similarity Score: {similarity_score:.2f}")
    similarity_score, y1, y2, sr = fitness_frequency_domain(file1, file2, tolerance = 10.0)
    print(f"Frequency Similarity Score: {similarity_score:.2f}")
    print("Playing trimmed audio1...")
    play_audio(y1, sr)
    print("Playing trimmed audio2...")
    play_audio(y2, sr)
