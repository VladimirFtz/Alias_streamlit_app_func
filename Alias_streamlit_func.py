import numpy as np
import pandas as pd 
#import matplotlib.pyplot as plt
#import seaborn as sns
#from datetime import datetime
import streamlit as st
from PIL import Image

from sentence_transformers import SentenceTransformer, util

#model = SentenceTransformer('all-MiniLM-L6-v2')

st.cache_data()
def load_model():
	  return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

alias_num = 7

# Счетываем данные после того как записали в таблицу нового человека и преобразуем их правильно
data = pd.read_csv('Alias_data_1.csv')
for i in range(len(data['embedding_1'])):
    for j in range(1, alias_num + 1):
        data[f'embedding_{j}'][i] = np.array(data[f'embedding_{j}'][i].replace('[', ' ').replace(']', ' ').split()).astype(np.float64)

st.title(':green[Родственные души и умы по Alias:) Alias soulmates ✨]')

image = Image.open('3.jpg')
st.image(image)

name = st.text_input('Введите ваше имя, дорогой игрок')
phone_number = st.text_input('Введите ваш номер телефона (нужен только для идентификакции, в случае повторения имен (без +7 и др))')
link = st.text_input('Добавьте ссылку, чтобы игроки смогли вас найти:) (ссылка на страницу в соц сети, телеграмм и др.)')

word_1 = 'Закладка'
word_2 = 'Опрометчивый'
word_3 = 'Мечта'
word_4 = 'Ёшкин кот'
word_5 = 'Смысловая галлюцинация'
word_6 = 'Авось'
word_7 = 'Любовь'

st.header(':green[Готовы к игре? Если да, то поехали! 🤘]')
st.header(':green[Напишите одним, ну или несколькими:) предложениями как бы вы объяснили следующие слова, не используя однокоренных слов!💫]')

image = Image.open('5.JPG')
st.image(image)

st.text('')
st.header(':blue[Закладка] ✌')
alias_1 = st.text_input('Закладка', label_visibility="hidden")


st.text('')
st.text('')
st.header(':blue[Опрометчивый] 😸')
alias_2 = st.text_input('Опрометчивый', label_visibility="hidden")

st.text('')
st.text('')
st.header(':blue[Мечта] ✨')
alias_3 = st.text_input('Прозрачность', label_visibility="hidden")

st.text('')
st.text('')
st.header(':blue[Ёшкин кот] 👀')
alias_4 = st.text_input('Ёшкин кот', label_visibility="hidden")

st.text('')
st.text('')
st.header(':blue[Смысловая галлюцинация] ☀')
alias_5 = st.text_input('Смысловая галлюцинация', label_visibility="hidden")

st.text('')
st.text('')
st.header(':blue[Авось] 🌿')
alias_6 = st.text_input('Авось', label_visibility="hidden")

st.text('')
st.text('')
st.header(':blue[Любовь] ❤')
alias_7 = st.text_input('Любовь', label_visibility="hidden")

add_and_find = st.button('Поехали, найдем кого нибудь')
if add_and_find:
    # Если номер телефона есть в таблице, перезаписываем строку, если нет добавляем новую.
    row_index = data.loc[data['phone_number'] == int(phone_number)].index
    
    #Рассчитываем embeddings
    embedding_1 = model.encode(alias_1)
    embedding_2 = model.encode(alias_2)
    embedding_3 = model.encode(alias_3)
    embedding_4 = model.encode(alias_4)
    embedding_5 = model.encode(alias_5)
    embedding_6 = model.encode(alias_6)
    embedding_7 = model.encode(alias_7)
    
    if row_index.size > 0:
        data.loc[row_index.to_list()[0], ['name', 'phone_number', 'link',
                             'alias_1', 'alias_2', 'alias_3','alias_4', 'alias_5', 'alias_6', 'alias_7',
                             'embedding_1', 'embedding_2', 'embedding_3','embedding_4','embedding_5',
                             'embedding_6','embedding_7',
                             'best_pair_1','best_pair_2','best_pair_3',
                             'best_score_1','best_score_2','best_score_3']] = \
        name, phone_number, link, alias_1, alias_2, alias_3, alias_4, alias_5, alias_6, alias_7, \
        embedding_1, embedding_2, embedding_3, embedding_4, embedding_5, embedding_6, embedding_7, 0, 0, 0, 0, 0, 0
        
    else:
        data.loc[len(data.index)] = [name, phone_number, link, alias_1, alias_2, alias_3, alias_4, alias_5, alias_6, alias_7, \
        embedding_1, embedding_2, embedding_3, embedding_4, embedding_5, embedding_6, embedding_7, 0, 0, 0, 0, 0, 0]
    #Записываем дату, важно записать без индекса
    data.to_csv('Alias_data_1.csv',index=False)
    
    # Счетываем данные после того как записали в таблицу нового человека и преобразуем их правильно
    data = pd.read_csv('Alias_data_1.csv')
    for i in range(len(data['embedding_1'])):
        for j in range(1, alias_num + 1):
            data[f'embedding_{j}'][i] = np.array(data[f'embedding_{j}'][i].replace('[', ' ').replace(']', ' ').split()).astype(np.float64)
    
    # Счетаем все средние значения близости предложений у всех пар пользователей
    scores_by_i = []
    strings_count, _ = data.shape
    for i in range(0,strings_count):
        inner_scores = []
        for j in range(0,strings_count):
            cosine_scores = []
            for k in range(1, alias_num + 1):
                cosine_scores.append(float(util.cos_sim(data.iloc[i][f'embedding_{k}'], data.iloc[j][f'embedding_{k}'])))
            inner_scores.append([i, j, np.mean(cosine_scores)])
        scores_by_i.append(inner_scores)
    
    # Ищем и записывем близких по смыслу пользователей
    scores_by_i = np.array(scores_by_i)
    for i in range(0,len(scores_by_i)):
        scores = scores_by_i[i]
        sorted_scores = scores[scores[:, 2].argsort()[::-1]]
        best_counter = 1
        k = 0
        while best_counter < 3:
            if sorted_scores[k,1] != sorted_scores[k,0]:
                data.loc[i,f'best_pair_{best_counter}'] = sorted_scores[k,1]
                data.loc[i,f'best_score_{best_counter}'] = sorted_scores[k,2]
                best_counter += 1
            k += 1
    
    #Записываем дату, важно записать без индекса
    data.to_csv('Alias_data_1.csv',index=False)
    
    # Выводим лучшие пары игрока
    current = data.loc[data['phone_number'] == int(phone_number)]
    results = pd.DataFrame(columns = ['Схожие умы', 'Схожесть мыслей', 'Контакт'])
    results.loc[len(results.index)] = [data.iloc[int(current['best_pair_1'])]['name'], float(current['best_score_1']),
                               data.iloc[int(current['best_pair_1'])]['link']]
    results.loc[len(results.index)] = [data.iloc[int(current['best_pair_2'])]['name'], float(current['best_score_2']),
                               data.iloc[int(current['best_pair_2'])]['link']]
    # Выводим схожие умы
    st.header(':green[А вот и игроки с похожими мыслями] 🧭')
    st.dataframe(results)
    
    st.header(':green[Объяснения ваши и похожих игроков] 👀')
    # Выводим предложения игрока и его лучших пар
    alias_1 = pd.DataFrame(columns = ['Схожие умы', word_1])
    alias_1.loc[len(alias_1.index)] = [str(current['name'].values[0]), str(current['alias_1'].values[0])]
    alias_1.loc[len(alias_1.index)] = [data.iloc[int(current['best_pair_1'])]['name'], data.iloc[int(current['best_pair_1'])]['alias_1']]
    alias_1.loc[len(alias_1.index)] = [data.iloc[int(current['best_pair_2'])]['name'], data.iloc[int(current['best_pair_2'])]['alias_1']]
    st.dataframe(alias_1)
    
    # Выводим предложения игрока и лучших пар
    alias_2 = pd.DataFrame(columns = ['Схожие умы', word_2])
    alias_2.loc[len(alias_2.index)] = [str(current['name'].values[0]), str(current['alias_2'].values[0])]
    alias_2.loc[len(alias_2.index)] = [data.iloc[int(current['best_pair_1'])]['name'], data.iloc[int(current['best_pair_1'])]['alias_2']]
    alias_2.loc[len(alias_2.index)] = [data.iloc[int(current['best_pair_2'])]['name'], data.iloc[int(current['best_pair_2'])]['alias_2']]
    st.dataframe(alias_2)

    # Выводим предложения игрока и лучших пар
    alias_3 = pd.DataFrame(columns = ['Схожие умы', word_3])
    alias_3.loc[len(alias_3.index)] = [str(current['name'].values[0]), str(current['alias_3'].values[0])]
    alias_3.loc[len(alias_3.index)] = [data.iloc[int(current['best_pair_1'])]['name'], data.iloc[int(current['best_pair_1'])]['alias_3']]
    alias_3.loc[len(alias_3.index)] = [data.iloc[int(current['best_pair_2'])]['name'], data.iloc[int(current['best_pair_2'])]['alias_3']]
    st.dataframe(alias_3)
    
    # Выводим предложения игрока и лучших пар
    alias_4 = pd.DataFrame(columns = ['Схожие умы', word_4])
    alias_4.loc[len(alias_4.index)] = [str(current['name'].values[0]), str(current['alias_4'].values[0])]
    alias_4.loc[len(alias_4.index)] = [data.iloc[int(current['best_pair_1'])]['name'], data.iloc[int(current['best_pair_1'])]['alias_4']]
    alias_4.loc[len(alias_4.index)] = [data.iloc[int(current['best_pair_2'])]['name'], data.iloc[int(current['best_pair_2'])]['alias_4']]
    st.dataframe(alias_4)
    
    # Выводим предложения игрока и лучших пар
    alias_5 = pd.DataFrame(columns = ['Схожие умы', word_5])
    alias_5.loc[len(alias_5.index)] = [str(current['name'].values[0]), str(current['alias_5'].values[0])]
    alias_5.loc[len(alias_5.index)] = [data.iloc[int(current['best_pair_1'])]['name'], data.iloc[int(current['best_pair_1'])]['alias_5']]
    alias_5.loc[len(alias_5.index)] = [data.iloc[int(current['best_pair_2'])]['name'], data.iloc[int(current['best_pair_2'])]['alias_5']]
    st.dataframe(alias_5)

    # Выводим предложения игрока и лучших пар
    alias_6 = pd.DataFrame(columns = ['Схожие умы', word_6])
    alias_6.loc[len(alias_6.index)] = [str(current['name'].values[0]), str(current['alias_6'].values[0])]
    alias_6.loc[len(alias_6.index)] = [data.iloc[int(current['best_pair_1'])]['name'], data.iloc[int(current['best_pair_1'])]['alias_6']]
    alias_6.loc[len(alias_6.index)] = [data.iloc[int(current['best_pair_2'])]['name'], data.iloc[int(current['best_pair_2'])]['alias_6']]
    st.dataframe(alias_6)
    
    # Выводим предложения игрока и лучших пар
    alias_7 = pd.DataFrame(columns = ['Схожие умы', word_7])
    alias_7.loc[len(alias_7.index)] = [str(current['name'].values[0]), str(current['alias_7'].values[0])]
    alias_7.loc[len(alias_7.index)] = [data.iloc[int(current['best_pair_1'])]['name'], data.iloc[int(current['best_pair_1'])]['alias_7']]
    alias_7.loc[len(alias_7.index)] = [data.iloc[int(current['best_pair_2'])]['name'], data.iloc[int(current['best_pair_2'])]['alias_7']]
    st.dataframe(alias_7)

st.text('')
st.header(':green[Если уже играли! И не хочешь снова, а хочешь глянуть результаты, то нажимай снизу 👇]')
st.subheader(':red[Обязательно введи номер телефона для поиска]')
phone_number = st.text_input('Введите ваш номер телефона (без +7 и др))')
watch_results = st.button('Посмотреть компаньонов 🧭')
if watch_results:
    st.header(':green[Игроки с похожими мыслями] 🧭')
    # Счетываем данные после того как записали в таблицу нового человека и преобразуем их правильно
    data = pd.read_csv('Alias_data_1.csv')
    for i in range(len(data['embedding_1'])):
        for j in range(1, alias_num + 1):
            data[f'embedding_{j}'][i] = np.array(data[f'embedding_{j}'][i].replace('[', ' ').replace(']', ' ').split()).astype(np.float64)
    
    # Счетаем все средние значения близости предложений у всех пар пользователей
    scores_by_i = []
    strings_count, _ = data.shape
    for i in range(0,strings_count):
        inner_scores = []
        for j in range(0,strings_count):
            cosine_scores = []
            for k in range(1, alias_num + 1):
                cosine_scores.append(float(util.cos_sim(data.iloc[i][f'embedding_{k}'], data.iloc[j][f'embedding_{k}'])))
            inner_scores.append([i, j, np.mean(cosine_scores)])
        scores_by_i.append(inner_scores)
    
    # Ищем и записывем близких по смыслу пользователей
    scores_by_i = np.array(scores_by_i)
    for i in range(0,len(scores_by_i)):
        scores = scores_by_i[i]
        sorted_scores = scores[scores[:, 2].argsort()[::-1]]
        best_counter = 1
        k = 0
        while best_counter < 3:
            if sorted_scores[k,1] != sorted_scores[k,0]:
                data.loc[i,f'best_pair_{best_counter}'] = sorted_scores[k,1]
                data.loc[i,f'best_score_{best_counter}'] = sorted_scores[k,2]
                best_counter += 1
            k += 1
            
    row_index = data.loc[data['phone_number'] == int(phone_number)].index
    current = data.loc[data['phone_number'] == int(phone_number)]
    if row_index.size > 0:
        # Выводим лучшие пары игрока
        results = pd.DataFrame(columns = ['Схожие умы', 'Схожесть мыслей', 'Контакт'])
        results.loc[len(results.index)] = [data.iloc[int(current['best_pair_1'])]['name'], float(current['best_score_1']),
                                   data.iloc[int(current['best_pair_1'])]['link']]
        results.loc[len(results.index)] = [data.iloc[int(current['best_pair_2'])]['name'], float(current['best_score_2']),
                                   data.iloc[int(current['best_pair_2'])]['link']]
        # Выводим схожие умы
        st.dataframe(results)
    else:
        st.header(':red[Введите телефон!]')
        

watch_sentences = st.button('Посмотреть объяснения компаньонов 👀')
if watch_sentences:
    data = pd.read_csv('Alias_data_1.csv')
    row_index = data.loc[data['phone_number'] == int(phone_number)].index
    current = data.loc[data['phone_number'] == int(phone_number)]
    if row_index.size > 0:
        st.header(':green[Объяснения ваши и похожих игроков] 👀')
        # Выводим предложения игрока и его лучших пар
        alias_1 = pd.DataFrame(columns = ['Схожие умы', word_1])
        alias_1.loc[len(alias_1.index)] = [str(current['name'].values[0]), str(current['alias_1'].values[0])]
        alias_1.loc[len(alias_1.index)] = [data.iloc[int(current['best_pair_1'])]['name'], data.iloc[int(current['best_pair_1'])]['alias_1']]
        alias_1.loc[len(alias_1.index)] = [data.iloc[int(current['best_pair_2'])]['name'], data.iloc[int(current['best_pair_2'])]['alias_1']]
        st.dataframe(alias_1)
        
        # Выводим предложения игрока и лучших пар
        alias_2 = pd.DataFrame(columns = ['Схожие умы', word_2])
        alias_2.loc[len(alias_2.index)] = [str(current['name'].values[0]), str(current['alias_2'].values[0])]
        alias_2.loc[len(alias_2.index)] = [data.iloc[int(current['best_pair_1'])]['name'], data.iloc[int(current['best_pair_1'])]['alias_2']]
        alias_2.loc[len(alias_2.index)] = [data.iloc[int(current['best_pair_2'])]['name'], data.iloc[int(current['best_pair_2'])]['alias_2']]
        st.dataframe(alias_2)

        # Выводим предложения игрока и лучших пар
        alias_3 = pd.DataFrame(columns = ['Схожие умы', word_3])
        alias_3.loc[len(alias_3.index)] = [str(current['name'].values[0]), str(current['alias_3'].values[0])]
        alias_3.loc[len(alias_3.index)] = [data.iloc[int(current['best_pair_1'])]['name'], data.iloc[int(current['best_pair_1'])]['alias_3']]
        alias_3.loc[len(alias_3.index)] = [data.iloc[int(current['best_pair_2'])]['name'], data.iloc[int(current['best_pair_2'])]['alias_3']]
        st.dataframe(alias_3)
        
        # Выводим предложения игрока и лучших пар
        alias_4 = pd.DataFrame(columns = ['Схожие умы', word_4])
        alias_4.loc[len(alias_4.index)] = [str(current['name'].values[0]), str(current['alias_4'].values[0])]
        alias_4.loc[len(alias_4.index)] = [data.iloc[int(current['best_pair_1'])]['name'], data.iloc[int(current['best_pair_1'])]['alias_4']]
        alias_4.loc[len(alias_4.index)] = [data.iloc[int(current['best_pair_2'])]['name'], data.iloc[int(current['best_pair_2'])]['alias_4']]
        st.dataframe(alias_4)
        
        # Выводим предложения игрока и лучших пар
        alias_5 = pd.DataFrame(columns = ['Схожие умы', word_5])
        alias_5.loc[len(alias_5.index)] = [str(current['name'].values[0]), str(current['alias_5'].values[0])]
        alias_5.loc[len(alias_5.index)] = [data.iloc[int(current['best_pair_1'])]['name'], data.iloc[int(current['best_pair_1'])]['alias_5']]
        alias_5.loc[len(alias_5.index)] = [data.iloc[int(current['best_pair_2'])]['name'], data.iloc[int(current['best_pair_2'])]['alias_5']]
        st.dataframe(alias_5)

        # Выводим предложения игрока и лучших пар
        alias_6 = pd.DataFrame(columns = ['Схожие умы', word_6])
        alias_6.loc[len(alias_6.index)] = [str(current['name'].values[0]), str(current['alias_6'].values[0])]
        alias_6.loc[len(alias_6.index)] = [data.iloc[int(current['best_pair_1'])]['name'], data.iloc[int(current['best_pair_1'])]['alias_6']]
        alias_6.loc[len(alias_6.index)] = [data.iloc[int(current['best_pair_2'])]['name'], data.iloc[int(current['best_pair_2'])]['alias_6']]
        st.dataframe(alias_6)
        
        # Выводим предложения игрока и лучших пар
        alias_7 = pd.DataFrame(columns = ['Схожие умы', word_7])
        alias_7.loc[len(alias_7.index)] = [str(current['name'].values[0]), str(current['alias_7'].values[0])]
        alias_7.loc[len(alias_7.index)] = [data.iloc[int(current['best_pair_1'])]['name'], data.iloc[int(current['best_pair_1'])]['alias_7']]
        alias_7.loc[len(alias_7.index)] = [data.iloc[int(current['best_pair_2'])]['name'], data.iloc[int(current['best_pair_2'])]['alias_7']]
        st.dataframe(alias_7)
    else:
        st.header(':red[Введите телефон!]')

watch_scores = st.button('Посмотреть схожесть объяснений со всеми игроками 🧭')
if watch_scores:
    # Счетываем данные после того как записали в таблицу нового человека и преобразуем их правильно
    data = pd.read_csv('Alias_data_1.csv')
    for i in range(len(data['embedding_1'])):
        for j in range(1, alias_num + 1):
            data[f'embedding_{j}'][i] = np.array(data[f'embedding_{j}'][i].replace('[', ' ').replace(']', ' ').split()).astype(np.float64)
    
    # Счетаем все средние значения близости предложений у всех пар пользователей
    scores_by_i = []
    strings_count, _ = data.shape
    for i in range(0,strings_count):
        inner_scores = []
        for j in range(0,strings_count):
            cosine_scores = []
            for k in range(1, alias_num + 1):
                cosine_scores.append(float(util.cos_sim(data.iloc[i][f'embedding_{k}'], data.iloc[j][f'embedding_{k}'])))
            inner_scores.append([i, j, np.mean(cosine_scores)])
        scores_by_i.append(inner_scores)
        
    row_index = data.loc[data['phone_number'] == int(phone_number)].index.to_list()[0]
    scores = np.array(scores_by_i[row_index])[:,2]
    player_scores = pd.DataFrame(columns=['name','Схожесть мыслей'])
    player_scores['name'] = data['name']
    player_scores['Схожесть мыслей'] = scores
    player_scores.sort_values(by='Схожесть мыслей', ascending=False, inplace=True)
    st.dataframe(player_scores)
    #fig = plt.figure(figsize=(10, 4))
    #plt.bar(player_scores['name'][1:],player_scores['Схожесть мыслей'][1:])
    #st.pyplot(fig)

st.text('')
st.text('')
st.text('')
st.text('')
st.text('')
st.text('')
st.text('')
watch_data = st.button('Посмотреть таблицу всех игроков')
if watch_data:
    st.dataframe(data)

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode('utf-8')

data_file = convert_df(data)

st.download_button(
    label="Скачать таблицу всех игроков",
    data=data_file,
    file_name='Alias_data_1_1.csv',
    mime='text/csv',
)
    