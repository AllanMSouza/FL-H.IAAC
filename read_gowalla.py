import numpy as np
import pandas as pd
import json
import math
import statistics as st
from sklearn.model_selection import KFold
from analysis.base_plots import bar_plot


def distance_importance(distance):
    distance_sigma = 10
    distance = distance * distance
    distance = -(distance / (distance_sigma * distance_sigma))
    distance = math.exp(distance)

    return distance


def duration_importance(duration):
    duration_sigma = 10
    duration = duration * duration
    duration = -(duration / (duration_sigma * duration_sigma))
    duration = math.exp(duration)

    return duration

def remove_hour_from_sequence_y(list_events: list):

    locations = []
    for e in list_events:
        locations.append(e[0])

    return locations

def sequence_to_x_y(list_events: list, step_size):
    x = []
    y = []

    for i in range(step_size, len(list_events)-1):
        e = list_events[i]
        step = list_events[i-step_size:i]
        x.append(step)
        y.append(list_events[i+1])

    return x, y

def _sequence_to_list(series):
    count = 0
    if "." in series:
        count += 1
    series = json.loads(series.replace("'", ""))
    return np.array(series)

def _add_total(user):

    total = []
    user = user.tolist()
    for i in range(len(user)):
        total.append(len(user[i]))

    return np.array(total)

file = "/home/claudio/Documentos/pycharm_projects/FL-H.IAAC/dataset_utils/data/Gowalla/raw data/gowalla_7_categories_sequences.csv"

df = pd.read_csv(file)

# 37828 clientes

df['sequence'] = df['sequence'].apply(lambda e: _sequence_to_list(e))
df['total'] = _add_total(df['sequence'])
np.random.seed(1)
df['rand'] = np.random.rand(len(df))
minimum = 400
n = 1000
df = df.sort_values(by='total', ascending=False).query("""total >= {}""".format(minimum))
df = df.head(n)

# 11300 clientes

df['id'] = np.array([i for i in range(len(df))])
base_dir = "/home/claudio/Documentos/pycharm_projects/FL-H.IAAC/dataset_utils/data/Gowalla/data/"
users_ids = df['id'].tolist()
sequences = df['sequence'].tolist()
categories_list = df['categories'].tolist()
x_list = []
y_list = []
countries = {}
max_country = 0
max_distance = 0
max_duration = 0
step_size = 3
n_splits = 5
ids_remove = []

distance_list = []
duration_list = []

weekd_ay_categories = {'Week day': [], 'Category': []}

count = 0
for i in range(len(users_ids)):

    user_id = users_ids[i]
    user_sequence = sequences[i]
    print("sequencia: ", len(user_sequence), user_sequence.shape, type(user_sequence))
    categories = json.loads(categories_list[i])
    new_sequence = []

    size = len(user_sequence)
    for j in range(size):
        location_category_id = user_sequence[j][0]
        hour = user_sequence[j][1]
        country = user_sequence[j][2]
        distance = user_sequence[j][3]
        duration = user_sequence[j][4]
        if j < len(user_sequence) -1:
            if duration > 72 and user_sequence[j + 1][4] > 72:
                continue
        week_day = user_sequence[j][5]
        poi_id = user_sequence[j][7]

        if distance > 50:
            distance = 50
        if duration > 72:
            duration = 72
        distance_list.append(distance)
        duration_list.append(duration)
        countries[country] = 0
        if country > max_country:
            max_country = country
        if distance > max_distance:
            max_distance = distance
        if duration > max_duration:
            max_duration = duration
        distance = distance_importance(distance)
        duration = duration_importance(duration)
        user_id = count
        weekd_ay_categories['Week day'].append(week_day)
        weekd_ay_categories['Category'].append(location_category_id)
        new_sequence.append([location_category_id, hour, country, distance, duration, week_day, user_id])

    x, y = sequence_to_x_y(new_sequence, step_size)
    y = remove_hour_from_sequence_y(y)

    user_df = pd.DataFrame({'x': x, 'y': y})
    user_df.to_csv("""{}{}.csv""".format(base_dir, count))
    x = user_df['x'].tolist()
    y = user_df['y'].tolist()

    x_list.append(x)
    y_list.append(y)

    count += 1

# print(len(x))
# exit()
#
# print("quantidade usuarios: ", len(users_ids))
# print("quantidade se: ", len(x_list))
# print("maior pais: ", max_country)
# print("maior distancia: ", max_distance)
# print("maior duracao: ", max_duration)
# print("distancia mediana: ", st.median(distance_list))
# print("duracao mediana: ", st.median(duration_list))
# print([len(i) for i in x_list])
# df['x'] = np.array(x_list)
# df['y'] = np.array(y_list)
# df = df[['id', 'x', 'y']]
#
# print("paises: ", len(list(countries.keys())))
# exit()
# # remove users that have few samples
# ids_remove_users = []
# ids_ = df['id'].tolist()
# x_list = df['x'].tolist()
# #x_list = [json.loads(x_list[i]) for i in range(len(x_list))]
# for i in range(df.shape[0]):
#     user = x_list[i]
#     if len(user) < n_splits or len(user) < int(minimum/step_size):
#         ids_remove_users.append(ids_[i])
#         continue
#
# # remove users that have few samples
# df = df[['id', 'x', 'y']].query("id not in " + str(ids_remove_users))
#
# x_users = df['x'].tolist()
# kf = KFold(n_splits=n_splits)
# users_train_indexes = [None] * n_splits
# users_test_indexes = [None] * n_splits
# for i in range(len(x_users)):
#     user = x_users[i]
#
#     j = 0
#
#     for train_indexes, test_indexes in kf.split(user):
#         if users_train_indexes[j] is None:
#             users_train_indexes[j] = [train_indexes]
#             users_test_indexes[j] = [test_indexes]
#         else:
#             users_train_indexes[j].append(train_indexes)
#             users_test_indexes[j].append(test_indexes)
#         j += 1
#
# print("treino", len(users_train_indexes))
# print("fold 0: ", len(users_train_indexes[0][1]), len(users_test_indexes[0][1]))
# print("fold 1: ", len(users_train_indexes[1][1]), len(users_test_indexes[1][1]))
# print("fold 2: ", len(users_train_indexes[2][1]), len(users_test_indexes[2][1]))
# print("fold 3: ", len(users_train_indexes[3][1]), len(users_test_indexes[3][1]))
# print("fold 4: ", len(users_train_indexes[4][1]), len(users_test_indexes[4][1]))
#
# max_userid = len(df)
# print("Quantidade de usuários: ", len(df))
# # update users id
# df['id'] = np.array([i for i in range(len(df))])
# ids_list = df['id'].tolist()
# x_list = x_users
# # print("ant")
# # print(x_list[0][0])
# # x_list = [json.loads(x_list[i]) for i in range(len(x_list))]
# y_list = df['y'].tolist()
# #y_list = [json.loads(y_list[i]) for i in range(len(y_list))]
# for i in range(len(x_list)):
#     sequences_list = x_list[i]
#
#
#     for j in range(len(sequences_list)):
#         user_sequence = sequences_list[j]
#         for k in range(len(user_sequence)):
#
#             user_sequence[k][-1] = ids_list[i]
#
#
#         sequences_list[j] = user_sequence
#     x_list[i] = sequences_list
#
# ids = df['id'].tolist()
# x = x_list
# y = y_list
#
# users_trajectories = df.to_numpy()
#
# print(df)
week_day_categoriees_df = pd.DataFrame(weekd_ay_categories)
week_day_categoriees_df = week_day_categoriees_df.groupby(['Week day', 'Category']).apply(lambda e:len(e)).reset_index()
week_day_categoriees_df['Count'] = np.array(week_day_categoriees_df[0].tolist())
week_day_categoriees_df['Count'] = np.array(week_day_categoriees_df['Count']) / np.array(week_day_categoriees_df['Count'].max())
print(week_day_categoriees_df)
bar_plot(week_day_categoriees_df, base_dir="", file_name='week_day_categories', x_column='Week day', y_column='Count', hue='Category', title='', )


