import pandas as pd
import mne
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

def extract_stimuli_events(markers_df, eeg_start_time, sfreq=128):
    """Извлекает события стимулов из markers_df по колонке marker_value"""
    
    # Список всех 24 стимулов из ТЗ
    stimulus_types = [
        'VK_JAPAN_INFO', 'VK_JAPAN_COM', 'VK_JAPAN_THR',
        'TG_JAPAN_INFO', 'TG_JAPAN_COM', 'TG_JAPAN_THR',
        'TG_MUSK_INFO', 'TG_MUSK_COM', 'TG_MUSK_THR',
        'VK_MUSK_INFO', 'VK_MUSK_COM', 'VK_MUSK_THR',
        'VK_BORISOV_INFO', 'VK_BORISOV_COM', 'VK_BORISOV_THR',
        'TG_BORISOV_INFO', 'TG_BORISOV_COM', 'TG_BORISOV_THR',
        'TG_EGE_INFO', 'TG_EGE_COM', 'TG_EGE_THR',
        'VK_EGE_INFO', 'VK_EGE_COM', 'VK_EGE_THR'
    ]
    
    # Фильтруем маркеры по колонке marker_value
    stimulus_markers = markers_df[markers_df['marker_value'].isin(stimulus_types)]
    
    print(f"Найдено маркеров стимулов: {len(stimulus_markers)}")
    print("Уникальные marker_value:", stimulus_markers['marker_value'].unique())
    
    events = []
    event_id = {}
    event_counter = 1
    
    for idx, row in stimulus_markers.iterrows():
        # Время стимула относительно начала EEG записи
        # latency - это время от начала записи в секундах
        stim_time = row['latency']  # уже относительно начала записи
        
        # Конвертируем в samples
        sample = int(stim_time * sfreq)
        
        # Проверяем, что время положительное
        if sample >= 0:
            stimulus_name = row['marker_value']
            events.append([sample, 0, event_counter])
            event_id[stimulus_name] = event_counter
            event_counter += 1
            
            print(f"Стимул: {stimulus_name}, время: {stim_time:.2f} сек, sample: {sample}")
    
    return np.array(events), event_id

def process_participant_erp(participant_id, data_folder):
    """Обрабатывает данные одного респондента для ERP анализа"""
    
    try:
        # Загрузка файлов
        eeg_file = f"{participant_id}_Восприятие инфоповодов_RAW.csv"
        marker_file = f"{participant_id}_Восприятие инфоповодов_intervalMarker.csv"
        
        eeg_df = pd.read_csv(Path(data_folder) / eeg_file, header=1)
        markers_df = pd.read_csv(Path(data_folder) / marker_file)
        
        # Определяем время начала EEG записи (первая временная метка)
        eeg_start_time = eeg_df['Timestamp'].iloc[0]
        
        # Выделяем ЭЭГ каналы
        eeg_channels = ['EEG.AF3', 'EEG.F7', 'EEG.F3', 'EEG.FC5', 'EEG.T7', 'EEG.P7', 
                       'EEG.O1', 'EEG.O2', 'EEG.P8', 'EEG.T8', 'EEG.FC6', 'EEG.F4', 
                       'EEG.F8', 'EEG.AF4']
        
        # Создание Raw объекта
        data = eeg_df[eeg_channels].values.T * 1e-6  # конвертируем в Вольты
        sfreq = 128  # частота Emotiv
        
        info = mne.create_info(ch_names=eeg_channels, sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(data, info)
        
        # Предобработка
        print(f"Применение фильтров для участника {participant_id}...")
        raw.filter(1, 30)  # band-pass фильтр
        raw.set_eeg_reference('average')  # редреференсирование
        
        # Извлекаем события из маркеров
        events, event_id = extract_stimuli_events(markers_df, eeg_start_time, sfreq)
        
        if len(events) == 0:
            print(f"Не найдено событий для участника {participant_id}")
            # Выведем все уникальные marker_value для отладки
            print("Все уникальные marker_value в файле:")
            print(markers_df['marker_value'].unique())
            return None
        
        print(f"Найдено {len(events)} событий для участника {participant_id}")
        
        # Создание эпох
        epochs = mne.Epochs(raw, events, event_id, tmin=-0.2, tmax=1.0,
                           baseline=(-0.2, 0), preload=True, reject=None)
        
        # Усреднение по каждому типу стимула
        evoked_dict = {}
        for stim_name, stim_id in event_id.items():
            if stim_id in epochs.events[:, 2]:
                evoked_dict[stim_name] = epochs[stim_name].average()
        
        print(f"Участник {participant_id}: успешно обработано {len(evoked_dict)} стимулов")
        return evoked_dict
        
    except Exception as e:
        print(f"Ошибка при обработке участника {participant_id}: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_all_participants_erp(data_folder):
    """Обрабатывает всех респондентов и возвращает усредненные ERP"""
    
    # Список всех респондентов
    participant_ids = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 
                      '11', '12', '13', '14', '15', '16']  # добавьте всех
    # participant_ids = ['1']  # добавьте всех
    
    all_evoked = {}
    
    # Обрабатываем каждого респондента
    for participant_id in participant_ids:
        print(f"\n{'='*50}")
        print(f"Обработка участника {participant_id}")
        print(f"{'='*50}")
        
        evoked_dict = process_participant_erp(participant_id, data_folder)
        if evoked_dict:
            all_evoked[participant_id] = evoked_dict
        else:
            print(f"Участник {participant_id} не обработан")
    
    # Grand average для каждого стимула
    grand_averages = {}
    
    # Получаем все уникальные стимулы
    all_stimuli = set()
    for participant_data in all_evoked.values():
        all_stimuli.update(participant_data.keys())
    
    print(f"\n{'='*50}")
    print(f"УСРЕДНЕНИЕ ПО ВСЕМ УЧАСТНИКАМ")
    print(f"{'='*50}")
    print(f"Всего уникальных стимулов: {len(all_stimuli)}")
    
    # Усредняем по всем респондентам для каждого стимула
    for stimulus in sorted(all_stimuli):
        evoked_list = []
        for participant_id, participant_data in all_evoked.items():
            if stimulus in participant_data:
                evoked_list.append(participant_data[stimulus])
        
        if evoked_list:
            grand_averages[stimulus] = mne.grand_average(evoked_list)
            print(f"Стимул {stimulus}: усреднено по {len(evoked_list)} участникам")
        else:
            print(f"Стимул {stimulus}: нет данных от участников")
    
    return grand_averages, all_evoked

def plot_erp_results(grand_averages, output_folder):
    """Строит 24 графика ERP для каждого стимула с цветными каналами"""
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Полный список стимулов из ТЗ
    stimulus_order = [
        'VK_JAPAN_INFO', 'VK_JAPAN_COM', 'VK_JAPAN_THR',
        'TG_JAPAN_INFO', 'TG_JAPAN_COM', 'TG_JAPAN_THR',
        'TG_MUSK_INFO', 'TG_MUSK_COM', 'TG_MUSK_THR',
        'VK_MUSK_INFO', 'VK_MUSK_COM', 'VK_MUSK_THR',
        'VK_BORISOV_INFO', 'VK_BORISOV_COM', 'VK_BORISOV_THR',
        'TG_BORISOV_INFO', 'TG_BORISOV_COM', 'TG_BORISOV_THR',
        'TG_EGE_INFO', 'TG_EGE_COM', 'TG_EGE_THR',
        'VK_EGE_INFO', 'VK_EGE_COM', 'VK_EGE_THR'
    ]
    
    print(f"\n{'='*50}")
    print(f"СОЗДАНИЕ ЦВЕТНЫХ ГРАФИКОВ ERP")
    print(f"{'='*50}")
    
    created_count = 0
    
    # Цветовая палитра для каналов
    channel_colors = plt.cm.tab20(np.linspace(0, 1, 14))
    
    for stimulus in stimulus_order:
        if stimulus in grand_averages:
            evoked = grand_averages[stimulus]
            
            # Создаем красивый график с цветными каналами
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Рисуем каждый канал своим цветом
            times = evoked.times
            for i, ch_name in enumerate(evoked.ch_names):
                data = evoked.data[i] * 1e6  # конвертируем в микроВольты для лучшего отображения
                ax.plot(times, data, 
                       color=channel_colors[i], 
                       linewidth=1.5, 
                       label=ch_name,
                       alpha=0.8)
            
            # Настраиваем график
            ax.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Стимул')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
            
            # Добавляем область бейзлайна
            ax.axvspan(-0.2, 0, color='gray', alpha=0.1, label='Бейзлайн')
            
            ax.set_xlabel('Время (секунды)', fontsize=12)
            ax.set_ylabel('Амплитуда (мкВ)', fontsize=12)
            ax.set_title(f'ERP - {stimulus}\n(Усреднено по всем участникам)', 
                        fontsize=14, fontweight='bold', pad=20)
            
            # Легенда
            ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=10)
            
            # Сетка
            ax.grid(True, alpha=0.3)
            
            # Сохраняем
            filename = f"ERP_{stimulus}.png"
            filepath = Path(output_folder) / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"✓ Сохранен: {filename}")
            created_count += 1
        else:
            print(f"✗ Пропущен: {stimulus} (нет данных)")
    
    return created_count

def create_summary_report(grand_averages, individual_results, output_folder, created_count):
    """Создает сводный отчет"""
    
    report_file = Path(output_folder) / "ERP_analysis_report.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("ОТЧЕТ ПО АНАЛИЗУ ERP\n")
        f.write("===================\n\n")
        
        f.write("СТАТИСТИКА ОБРАБОТКИ:\n")
        f.write(f"- Участников обработано: {len(individual_results)}\n")
        f.write(f"- Стимулов найдено: {len(grand_averages)}\n")
        f.write(f"- Графиков создано: {created_count}\n")
        f.write(f"- Целевое количество графиков: 24\n\n")
        
        f.write("ОБРАБОТАННЫЕ СТИМУЛЫ:\n")
        for stimulus in sorted(grand_averages.keys()):
            # Считаем сколько участников имеют этот стимул
            participant_count = sum(1 for data in individual_results.values() if stimulus in data)
            f.write(f"- {stimulus} ({participant_count} участников)\n")
        
        # Отсутствующие стимулы
        all_stimuli = [
            'VK_JAPAN_INFO', 'VK_JAPAN_COM', 'VK_JAPAN_THR',
            'TG_JAPAN_INFO', 'TG_JAPAN_COM', 'TG_JAPAN_THR',
            'TG_MUSK_INFO', 'TG_MUSK_COM', 'TG_MUSK_THR',
            'VK_MUSK_INFO', 'VK_MUSK_COM', 'VK_MUSK_THR',
            'VK_BORISOV_INFO', 'VK_BORISOV_COM', 'VK_BORISOV_THR',
            'TG_BORISOV_INFO', 'TG_BORISOV_COM', 'TG_BORISOV_THR',
            'TG_EGE_INFO', 'TG_EGE_COM', 'TG_EGE_THR',
            'VK_EGE_INFO', 'VK_EGE_COM', 'VK_EGE_THR'
        ]
        
        missing = set(all_stimuli) - set(grand_averages.keys())
        if missing:
            f.write(f"\nОТСУТСТВУЮЩИЕ СТИМУЛЫ:\n")
            for stim in sorted(missing):
                f.write(f"- {stim}\n")

# Основной скрипт
if __name__ == "__main__":
    data_folder = "condition/data"
    output_folder = "erp_results"
    
    print(f"{'='*60}")
    print("НАЧАЛО ОБРАБОТКИ ERP ДАННЫХ")
    print(f"{'='*60}")
    
    # Обрабатываем всех респондентов
    grand_averages, individual_results = process_all_participants_erp(data_folder)
    
    print(f"\n{'='*60}")
    print("РЕЗУЛЬТАТЫ ОБРАБОТКИ")
    print(f"{'='*60}")
    print(f"Участников с данными: {len(individual_results)}")
    print(f"Уникальных стимулов: {len(grand_averages)}")
    
    # Строим графики
    created_count = plot_erp_results(grand_averages, output_folder)
    
    # Создаем отчет
    create_summary_report(grand_averages, individual_results, output_folder, created_count)
    
    print(f"\n{'='*60}")
    print("ОБРАБОТКА ЗАВЕРШЕНА!")
    print(f"{'='*60}")
    print(f"Создано графиков: {created_count}/24")
    print(f"Результаты сохранены в папку: {output_folder}")
    print(f"Отчет: {output_folder}/ERP_analysis_report.txt")