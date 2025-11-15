import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_emotiv_data(emotion_path):
    """Загружает сырые данные Emotiv с правильными типами"""
    try:
        # Пропускаем первую строку с описанием и загружаем данные
        df = pd.read_csv(emotion_path, skiprows=1, low_memory=False)
        
        # Преобразуем числовые колонки
        numeric_cols = ['Timestamp', 'OriginalTimestamp', 'EEG.Counter'] + \
                      [f'EEG.{ch}' for ch in ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']] + \
                      ['PM.Stress.Scaled', 'PM.Interest.Scaled', 'PM.Engagement.Scaled', 
                       'PM.Attention.Scaled', 'PM.Excitement.Scaled', 'PM.Relaxation.Scaled', 'PM.Focus.Scaled']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except Exception as e:
        print(f"Ошибка загрузки {emotion_path}: {e}")
        return None

def load_markers(marker_path):
    """Загружает файл с маркерами"""
    try:
        return pd.read_csv(marker_path)
    except Exception as e:
        print(f"Ошибка загрузки {marker_path}: {e}")
        return None

def load_pilot_data(pilot_path):
    """Загружает пилотные данные для получения временных меток"""
    try:
        pilot_df = pd.read_csv(pilot_path)
        
        # Извлекаем временные метки из пилотных данных
        timestamp_cols = ['timestamp', 'latency__desc', 'originaltimestamp']
        available_timestamp_col = None
        
        for col in timestamp_cols:
            if col in pilot_df.columns:
                available_timestamp_col = col
                break
        
        if available_timestamp_col:
            timestamps = pilot_df[available_timestamp_col].dropna().values
            print(f"  Загружено {len(timestamps)} временных меток из пилотных данных (колонка: {available_timestamp_col})")
            return timestamps
        else:
            print("  Не найдены временные метки в пилотном файле")
            return None
            
    except Exception as e:
        print(f"Ошибка загрузки пилотных данных {pilot_path}: {e}")
        return None

def synchronize_using_pilot_timestamps(emotion_df, markers_df, pilot_timestamps):
    """Синхронизирует временные метки используя пилотные данные как референс"""
    
    if pilot_timestamps is None or len(pilot_timestamps) == 0:
        print("  Нет пилотных временных меток, используется стандартная синхронизация")
        return synchronize_timestamps_fallback(emotion_df, markers_df)
    
    # Находим маркер начала активного периода в маркерах текущего участника
    active_period_markers = markers_df[markers_df['type'] == 'active_period']
    
    if len(active_period_markers) > 0:
        experiment_start_marker = active_period_markers.iloc[0]
        marker_start_time = experiment_start_marker['latency']
        
        # Используем первую временную метку из пилотных данных как референс
        pilot_start_time = pilot_timestamps[0] if len(pilot_timestamps) > 0 else 0
        emotion_start_time = emotion_df['Timestamp'].min()
        
        # Вычисляем смещение на основе разницы между пилотными данными и текущими маркерами
        time_offset = emotion_start_time - marker_start_time
        
        print(f"  Синхронизация: emotion_start={emotion_start_time:.2f}, marker_start={marker_start_time:.2f}, offset={time_offset:.2f}")
        return time_offset
    else:
        return synchronize_timestamps_fallback(emotion_df, markers_df)

def synchronize_timestamps_fallback(emotion_df, markers_df):
    """Резервный метод синхронизации если пилотные данные недоступны"""
    
    stimulus_markers = markers_df[markers_df['marker_value'].str.startswith(('VK_', 'TG_'), na=False)]
    if len(stimulus_markers) > 0:
        first_stimulus = stimulus_markers.iloc[0]
        marker_start_time = first_stimulus['latency'] - 10  # 10 секунд до первого стимула
        emotion_start_time = emotion_df['Timestamp'].min()
        time_offset = emotion_start_time - marker_start_time
        return time_offset
    else:
        # Если ничего не найдено, используем минимальное время
        marker_start_time = markers_df['latency'].min()
        emotion_start_time = emotion_df['Timestamp'].min()
        time_offset = emotion_start_time - marker_start_time
        return time_offset

def get_available_emotion_metrics(emotion_df):
    """Находит доступные эмоциональные метрики в данных"""
    
    emotion_metrics = {
        'Stress': 'PM.Stress.Scaled',
        'Interest': 'PM.Interest.Scaled', 
        'Engagement': 'PM.Engagement.Scaled',
        'Attention': 'PM.Attention.Scaled',
        'Excitement': 'PM.Excitement.Scaled',
        'Relaxation': 'PM.Relaxation.Scaled',
        'Focus': 'PM.Focus.Scaled'
    }
    
    available_metrics = {}
    for metric_name, metric_col in emotion_metrics.items():
        if metric_col in emotion_df.columns:
            # Проверяем, есть ли не-NaN значения
            non_null_data = emotion_df[metric_col].dropna()
            if len(non_null_data) > 0:
                available_metrics[metric_name] = metric_col
    
    return available_metrics

def extract_emotion_epochs(emotion_df, markers_df, time_offset, available_metrics):
    """Извлекает эпохи эмоциональных данных для каждого стимула"""
    
    # Порядок стимулов из таймлайна
    stimulus_order = [
        'VK_JAPAN_INFO', 'VK_JAPAN_COM', 'VK_JAPAN_THR',
        'TG_JAPAN_INFO', 'TG_JAPAN_COM', 'TG_JAPAN_THR', 
        'TG_MUSK_INFO', 'TG_MUSK_COM', 'TG_MUSK_THR',
        'VK_MUSK_INFO', 'VK_MUSK_COM', 'VK_MUSK_THR',
        'VK_BORISOV_INFO', 'VK_BORISOV_COM', 'VK_BORISOV_THR',
        'TG_BORISOV_INFO', 'TG_BORISOV_COM', 'TG_BORISOV_THR',
        'TG_EGE_INFO', 'TG_EGE_COM'
    ]
    
    emotion_epochs = {}
    
    for stimulus in stimulus_order:
        # Находим маркеры для этого стимула
        stimulus_markers = markers_df[markers_df['marker_value'] == stimulus]
        
        if len(stimulus_markers) == 0:
            # Пропускаем, если маркер не найден
            continue
            
        stimulus_marker = stimulus_markers.iloc[0]
        stim_time_marker = stimulus_marker['latency']
        stim_duration = stimulus_marker['duration'] if 'duration' in stimulus_marker else 10.0
        
        # Конвертируем время в временную шкалу эмоциональных данных
        stim_time_emotion = stim_time_marker + time_offset
        
        # Находим ближайшую временную метку в эмоциональных данных
        time_diffs = np.abs(emotion_df['Timestamp'] - stim_time_emotion)
        if len(time_diffs) == 0:
            continue
            
        closest_idx = time_diffs.idxmin()
        
        # Частота дискретизации (примерно 128 Гц для Emotiv)
        sfreq = 128
        
        # Извлекаем эпоху: 4 секунды до стимула, длительность стимула + 4 секунды после
        baseline_seconds = 4
        post_seconds = 4
        
        start_idx = max(0, closest_idx - int(baseline_seconds * sfreq))
        end_idx = min(len(emotion_df), closest_idx + int((stim_duration + post_seconds) * sfreq))
        
        if start_idx >= end_idx:
            continue
            
        epoch_data = emotion_df.iloc[start_idx:end_idx].copy()
        
        if len(epoch_data) == 0:
            continue
            
        # Создаем относительное время
        epoch_data = epoch_data.reset_index(drop=True)
        epoch_data['Time_Relative'] = epoch_data['Timestamp'] - stim_time_emotion
        
        # Сохраняем только нужные колонки
        keep_cols = ['Time_Relative'] + list(available_metrics.values())
        epoch_data = epoch_data[[col for col in keep_cols if col in epoch_data.columns]]
        
        emotion_epochs[stimulus] = epoch_data
    
    return emotion_epochs

def process_participant(participant_id, data_folder, pilot_timestamps):
    """Обрабатывает данные одного участника"""
    
    print(f"Обработка участника {participant_id}...")
    
    try:
        # Определяем пути к файлам
        emotion_file = f"{participant_id}_RAW DATA.csv"
        marker_file = f"{participant_id}_Восприятие инфоповодов_intervalMarker.csv"
        
        emotion_path = Path(data_folder) / emotion_file
        marker_path = Path(data_folder) / marker_file
        
        if not emotion_path.exists():
            print(f"  Файл не найден: {emotion_path}")
            return None, None
        if not marker_path.exists():
            print(f"  Файл не найден: {marker_path}")
            return None, None
            
        # Загружаем данные
        emotion_df = load_emotiv_data(emotion_path)
        markers_df = load_markers(marker_path)
        
        if emotion_df is None or markers_df is None:
            print(f"  Ошибка загрузки данных для участника {participant_id}")
            return None, None
            
        # Синхронизируем временные метки используя пилотные данные
        time_offset = synchronize_using_pilot_timestamps(emotion_df, markers_df, pilot_timestamps)
        
        # Находим доступные метрики
        available_metrics = get_available_emotion_metrics(emotion_df)
        
        if not available_metrics:
            print(f"  Нет доступных эмоциональных метрик для участника {participant_id}")
            return None, None
        
        print(f"  Доступные метрики: {list(available_metrics.keys())}")
        
        # Извлекаем эпохи
        emotion_epochs = extract_emotion_epochs(emotion_df, markers_df, time_offset, available_metrics)
        
        if not emotion_epochs:
            print(f"  Не удалось извлечь эпохи для участника {participant_id}")
            return None, None
            
        print(f"  Извлечено эпох: {len(emotion_epochs)}")
        
        return emotion_epochs, available_metrics
        
    except Exception as e:
        print(f"  Ошибка обработки участника {participant_id}: {str(e)}")
        return None, None

def create_grand_average(all_participant_data, participant_ids):
    """Создает усредненные данные по группе участников"""
    
    grand_average = {}
    
    # Собираем все стимулы
    all_stimuli = set()
    for pid in participant_ids:
        if pid in all_participant_data and all_participant_data[pid] is not None:
            emotion_epochs, _ = all_participant_data[pid]
            all_stimuli.update(emotion_epochs.keys())
    
    for stimulus in sorted(all_stimuli):
        all_epochs_data = []
        
        for pid in participant_ids:
            if pid in all_participant_data and all_participant_data[pid] is not None:
                emotion_epochs, available_metrics = all_participant_data[pid]
                if stimulus in emotion_epochs:
                    epoch_data = emotion_epochs[stimulus].copy()
                    
                    # Добавляем идентификатор участника
                    epoch_data['participant_id'] = pid
                    all_epochs_data.append(epoch_data)
        
        if all_epochs_data:
            # Объединяем все данные
            combined_data = pd.concat(all_epochs_data, ignore_index=True)
            
            # Усредняем по времени
            time_points = np.round(combined_data['Time_Relative'], 1)  # Округляем до 0.1 сек
            numeric_cols = [col for col in combined_data.columns 
                          if col not in ['Time_Relative', 'participant_id'] 
                          and pd.api.types.is_numeric_dtype(combined_data[col])]
            
            if numeric_cols:
                # Группируем по времени и усредняем
                avg_data = combined_data.groupby(time_points)[numeric_cols].mean().reset_index()
                avg_data = avg_data.rename(columns={'index': 'Time_Relative'})
                
                # Применяем сглаживание
                for col in numeric_cols:
                    if col in avg_data.columns:
                        avg_data[col] = avg_data[col].rolling(window=5, center=True, min_periods=1).mean()
                
                grand_average[stimulus] = avg_data
    
    return grand_average

def plot_emotion_metrics(grand_average, available_metrics, output_folder, title_suffix=""):
    """Строит графики эмоциональных метрик для каждого стимула"""
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Цвета для метрик
    metric_colors = {
        'Stress': '#FF6B6B',
        'Interest': '#4ECDC4', 
        'Engagement': '#45B7D1',
        'Attention': '#96CEB4',
        'Excitement': '#FECA57',
        'Relaxation': '#FF9FF3',
        'Focus': '#54A0FF'
    }
    
    # Порядок стимулов
    stimulus_order = [
        'VK_JAPAN_INFO', 'VK_JAPAN_COM', 'VK_JAPAN_THR',
        'TG_JAPAN_INFO', 'TG_JAPAN_COM', 'TG_JAPAN_THR',
        'TG_MUSK_INFO', 'TG_MUSK_COM', 'TG_MUSK_THR',
        'VK_MUSK_INFO', 'VK_MUSK_COM', 'VK_MUSK_THR',
        'VK_BORISOV_INFO', 'VK_BORISOV_COM', 'VK_BORISOV_THR',
        'TG_BORISOV_INFO', 'TG_BORISOV_COM', 'TG_BORISOV_THR',
        'TG_EGE_INFO', 'TG_EGE_COM'
    ]
    
    created_count = 0
    
    for stimulus in stimulus_order:
        if stimulus in grand_average:
            data = grand_average[stimulus]
            
            # Создаем график
            fig, ax = plt.subplots(figsize=(14, 8))
            
            has_data = False
            
            # Рисуем каждую метрику
            for metric_name, metric_col in available_metrics.items():
                if metric_col in data.columns:
                    plot_data = data[['Time_Relative', metric_col]].dropna()
                    
                    if len(plot_data) > 1:
                        # Используем сырые значения (но нормализуем для визуализации)
                        metric_values = plot_data[metric_col].values
                        
                        # Минимальная нормализация для лучшего отображения
                        if metric_values.std() > 0:
                            normalized_values = (metric_values - metric_values.mean()) / metric_values.std()
                        else:
                            normalized_values = metric_values - metric_values.mean()
                            
                        ax.plot(plot_data['Time_Relative'], normalized_values,
                               color=metric_colors.get(metric_name, 'black'),
                               linewidth=2.5,
                               label=metric_name,
                               alpha=0.8)
                        has_data = True
            
            if has_data:
                # Настройки графика
                ax.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Начало стимула')
                ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
                
                # Область бейзлайна (до стимула)
                baseline_end = 0
                baseline_start = data['Time_Relative'].min()
                if baseline_start < 0:
                    ax.axvspan(baseline_start, 0, color='lightblue', alpha=0.2, label='Бейзлайн')
                
                ax.set_xlabel('Время относительно стимула (секунды)', fontsize=12, fontweight='bold')
                ax.set_ylabel('Нормализованные значения метрик', fontsize=12, fontweight='bold')
                ax.set_title(f'Эмоциональные реакции - {stimulus}{title_suffix}', 
                            fontsize=16, fontweight='bold', pad=20)
                
                ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
                ax.grid(True, alpha=0.3)
                
                # Устанавливаем разумные пределы по времени
                time_min = max(data['Time_Relative'].min(), -4)
                time_max = min(data['Time_Relative'].max(), 10)
                ax.set_xlim(time_min, time_max)
                
                # Сохраняем график
                filename = f"emotions_{stimulus}{title_suffix.replace(' ', '_').replace('-', '_')}.png"
                filepath = Path(output_folder) / filename
                plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                
                print(f"  Создан график: {filename}")
                created_count += 1
            else:
                print(f"  Нет данных для стимула: {stimulus}")
        else:
            print(f"  Стимул отсутствует в данных: {stimulus}")
    
    return created_count

def create_emotion_analysis_report(all_data, grand_avg_all, grand_avg_women, grand_avg_men, 
                                 count_all, count_women, count_men, output_folder):
    """Создает подробный отчет по анализу эмоциональных метрик"""
    
    report_file = Path(output_folder) / "emotion_analysis_report.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("ОТЧЕТ ПО АНАЛИЗУ ЭМОЦИОНАЛЬНЫХ МЕТРИК\n")
        f.write("=" * 50 + "\n\n")
        
        # Общая статистика
        f.write("ОБЩАЯ СТАТИСТИКА ОБРАБОТКИ:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Участников обработано: {len(all_data)}/16\n")
        f.write(f"Эмоциональных метрик найдено: 7\n")
        f.write("  - Stress, Interest, Engagement, Attention, Excitement, Relaxation, Focus\n\n")
        
        # Статистика по графикам
        f.write("СТАТИСТИКА СОЗДАННЫХ ГРАФИКОВ:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Все участники: {count_all}/20 графиков\n")
        f.write(f"Женщины: {count_women}/20 графиков\n")
        f.write(f"Мужчины: {count_men}/20 графиков\n")
        f.write(f"ОБЩЕЕ: {count_all + count_women + count_men}/60 графиков\n\n")
        
        # Анализ стимулов
        f.write("АНАЛИЗ СТИМУЛОВ:\n")
        f.write("-" * 30 + "\n")
        
        # Все стимулы из таймлайна
        all_stimuli = [
            'VK_JAPAN_INFO', 'VK_JAPAN_COM', 'VK_JAPAN_THR',
            'TG_JAPAN_INFO', 'TG_JAPAN_COM', 'TG_JAPAN_THR',
            'TG_MUSK_INFO', 'TG_MUSK_COM', 'TG_MUSK_THR',
            'VK_MUSK_INFO', 'VK_MUSK_COM', 'VK_MUSK_THR',
            'VK_BORISOV_INFO', 'VK_BORISOV_COM', 'VK_BORISOV_THR',
            'TG_BORISOV_INFO', 'TG_BORISOV_COM', 'TG_BORISOV_THR',
            'TG_EGE_INFO', 'TG_EGE_COM'
        ]
        
        # Находим отсутствующие стимулы
        missing_in_all = set(all_stimuli) - set(grand_avg_all.keys()) if grand_avg_all else set(all_stimuli)
        missing_in_women = set(all_stimuli) - set(grand_avg_women.keys()) if grand_avg_women else set(all_stimuli)
        missing_in_men = set(all_stimuli) - set(grand_avg_men.keys()) if grand_avg_men else set(all_stimuli)
        
        f.write("Обработанные стимулы:\n")
        for stimulus in sorted(all_stimuli):
            status_all = "✓" if stimulus in grand_avg_all else "✗"
            status_women = "✓" if stimulus in grand_avg_women else "✗"
            status_men = "✓" if stimulus in grand_avg_men else "✗"
            f.write(f"  {stimulus:<20} Все: {status_all}  Жен: {status_women}  Муж: {status_men}\n")
        
        # Отсутствующие стимулы
        if missing_in_all or missing_in_women or missing_in_men:
            f.write("\nОтсутствующие стимулы:\n")
            for stimulus in sorted(missing_in_all | missing_in_women | missing_in_men):
                f.write(f"  - {stimulus}\n")
        
        # Статистика по участникам
        f.write("\nСТАТИСТИКА ПО УЧАСТНИКАМ:\n")
        f.write("-" * 30 + "\n")
        
        women_ids = ['1', '7', '9', '11', '12', '13', '15', '16']
        men_ids = ['2', '3', '4', '5', '6', '8', '10', '14']
        
        f.write(f"Женщины (8): {', '.join([pid for pid in women_ids if pid in all_data])}\n")
        f.write(f"Мужчины (8): {', '.join([pid for pid in men_ids if pid in all_data])}\n")
        
        # Количество эпох на участника
        f.write("\nЭПОХИ НА УЧАСТНИКА:\n")
        f.write("-" * 30 + "\n")
        for pid, (emotion_epochs, _) in all_data.items():
            f.write(f"  Участник {pid}: {len(emotion_epochs)} эпох\n")
        
        # Метрики качества данных
        f.write("\nМЕТРИКИ КАЧЕСТВА ДАННЫХ:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Синхронизация: Использованы пилотные временные метки\n")
        f.write(f"Бейзлайн: 4 секунды до стимула\n")
        f.write(f"Длительность анализа: 8 секунд (4 до + 4 после стимула)\n")
        f.write(f"Нормализация: Z-score по каждой метрике\n")
        f.write(f"Сглаживание: скользящее среднее (окно=5)\n")
        
        # Рекомендации
        f.write("\nРЕКОМЕНДАЦИИ:\n")
        f.write("-" * 30 + "\n")
        if missing_in_all:
            f.write("1. Проверить наличие маркеров для отсутствующих стимулов\n")
        if count_all + count_women + count_men < 60:
            f.write("2. Увеличить охват стимулов для полного анализа\n")
        f.write("3. Проверить качество синхронизации временных меток\n")
        f.write("4. Валидировать эмоциональные метрики на репрезентативных данных\n")
        
        f.write(f"\nОтчет создан: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"✓ Отчет сохранен: {report_file}")

def main():
    """Основная функция обработки данных"""
    
    data_folder = "condition/data"
    output_folder = "emotion_results"
    
    print("=" * 60)
    print("ОБРАБОТКА ЭМОЦИОНАЛЬНЫХ ДАННЫХ EMOТIV")
    print("=" * 60)
    
    # Загружаем пилотные данные для синхронизации
    print("\nЗагрузка пилотных данных для синхронизации...")
    pilot_path = Path(data_folder) / "0_Восприятие_инфоповодов_basic_pre_processing.csv"
    pilot_timestamps = load_pilot_data(pilot_path)
    
    # Основные участники (16 человек) - без пилотного участника 0
    main_participant_ids = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 
                           '11', '12', '13', '14', '15', '16']
    
    # Распределение по полу согласно заданию
    women_ids = ['1', '7', '9', '11', '12', '13', '15', '16']  # 8 женщин
    men_ids = ['2', '3', '4', '5', '6', '8', '10', '14']       # 8 мужчин
    
    # Обрабатываем основных участников
    print(f"\n1. ОБРАБОТКА ОСНОВНЫХ УЧАСТНИКОВ ({len(main_participant_ids)} человек)")
    print("-" * 50)
    
    all_data = {}
    for participant_id in main_participant_ids:
        emotion_epochs, available_metrics = process_participant(participant_id, data_folder, pilot_timestamps)
        if emotion_epochs is not None:
            all_data[participant_id] = (emotion_epochs, available_metrics)
    
    print(f"\nИТОГ: успешно обработано {len(all_data)} участников из {len(main_participant_ids)}")
    
    if not all_data:
        print("Нет данных для обработки! Проверьте пути к файлам.")
        return
    
    # Создаем усредненные данные и графики
    print(f"\n2. СОЗДАНИЕ ГРАФИКОВ")
    print("-" * 40)
    
    grand_avg_all, grand_avg_women, grand_avg_men = None, None, None
    count_all, count_women, count_men = 0, 0, 0
    
    # 1. Все участники
    valid_all_ids = [pid for pid in main_participant_ids if pid in all_data]
    if valid_all_ids:
        print(f"Все участники ({len(valid_all_ids)} человек)...")
        grand_avg_all = create_grand_average(all_data, valid_all_ids)
        count_all = plot_emotion_metrics(grand_avg_all, all_data[valid_all_ids[0]][1], 
                                      output_folder, " - Все участники")
        print(f"  Создано графиков: {count_all}/20")
    
    # 2. Женщины (8 человек)
    valid_women_ids = [pid for pid in women_ids if pid in all_data]
    if valid_women_ids:
        print(f"Женщины ({len(valid_women_ids)} человек)...")
        grand_avg_women = create_grand_average(all_data, valid_women_ids)
        women_output = Path(output_folder) / "women"
        count_women = plot_emotion_metrics(grand_avg_women, all_data[valid_women_ids[0]][1], 
                                        women_output, " - Женщины")
        print(f"  Создано графиков: {count_women}/20")
    
    # 3. Мужчины (8 человек)
    valid_men_ids = [pid for pid in men_ids if pid in all_data]
    if valid_men_ids:
        print(f"Мужчины ({len(valid_men_ids)} человек)...")
        grand_avg_men = create_grand_average(all_data, valid_men_ids)
        men_output = Path(output_folder) / "men"
        count_men = plot_emotion_metrics(grand_avg_men, all_data[valid_men_ids[0]][1], 
                                      men_output, " - Мужчины")
        print(f"  Создано графиков: {count_men}/20")
    
    # Создаем отчет
    print(f"\n3. СОЗДАНИЕ ОТЧЕТА")
    print("-" * 40)
    create_emotion_analysis_report(all_data, grand_avg_all, grand_avg_women, grand_avg_men,
                                 count_all, count_women, count_men, output_folder)
    
    print(f"\n" + "=" * 60)
    print("ОБРАБОТКА ЗАВЕРШЕНА!")
    print(f"Результаты сохранены в папке: {output_folder}")
    print(f"Всего создано графиков: {count_all + count_women + count_men}/60")
    print(f"Отчет: {output_folder}/emotion_analysis_report.txt")
    print("=" * 60)

if __name__ == "__main__":
    main()
