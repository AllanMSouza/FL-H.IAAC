from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from optparse import OptionParser
from base_plots import bar_plot, line_plot
from pathlib import Path
import os
import ast

class Verify:

    def start(self):

        self.base_dir = "output/exp_decaying/"

        if not Path(self.base_dir).exists():
            os.makedirs(self.base_dir+"csv/")
            os.makedirs(self.base_dir + "png/")
            os.makedirs(self.base_dir + "svg/")

        self.server_analysis()

    def curve(self, server_round, rounds_without_fit, start_round):
        # 0
        # max_rounds_without_fit = 3
        # alpha = 1.2
        # beta = 9
        # # normalizar dentro de 0 e 1
        # rounds_without_fit  = pow(min(rounds_without_fit, max_rounds_without_fit)/max_rounds_without_fit, alpha)
        # global_model_weight = 1
        # if rounds_without_fit > 0:
        # 	# o denominador faz com que a curva se prolongue com menor decaimento
        # 	# Quanto mais demorada for a convergência do modelo, maior deve ser o valor do denominador
        # 	eq1 = (-rounds_without_fit-(server_round)/beta)
        # 	# eq2: se divide por "rounds_without_fit" porque quanto mais rodadas sem treinamento, maior deve ser o peso
        # 	# do modelo global
        # 	eq2 = pow(2.7, eq1)
        # 	eq3 = min(eq2, 1)
        # 	global_model_weight = eq3
        # 1
        # max_rounds_without_fit = 3
        # alpha = 2
        # beta = 9
        # # normalizar dentro de 0 e 1
        # rounds_without_fit = pow(min(rounds_without_fit + 0.00001, max_rounds_without_fit) / (max_rounds_without_fit + 0.00001), -alpha)
        # global_model_weight = 1
        # if rounds_without_fit > 0:
        #     # o denominador faz com que a curva se prolongue com menor decaimento
        #     # Quanto mais demorada for a convergência do modelo, maior deve ser o valor do denominador
        #     eq1 = (- rounds_without_fit - (server_round - start_round) / beta)
        #     # eq2: se divide por "rounds_without_fit" porque quanto mais rodadas sem treinamento, maior deve ser o peso
        #     # do modelo global
        #     eq2 = np.exp(eq1)
        #     eq3 = min(eq2, 1)
        #     global_model_weight = eq3
        # 3
        # max_rounds_without_fit = 3
        # alpha = 1.2
        # beta = 9
        # start_round = 0
        # # normalizar dentro de 0 e 1
        # rounds_without_fit = pow(
        #     min(rounds_without_fit + 0.0001, max_rounds_without_fit), -alpha)
        # global_model_weight = 1
        # if rounds_without_fit > 0:
        #     # o denominador faz com que a curva se prolongue com menor decaimento
        #     # Quanto mais demorada for a convergência do modelo, maior deve ser o valor do denominador
        #     eq1 = (-rounds_without_fit - (server_round - start_round) / beta)
        #     # eq2: se divide por "rounds_without_fit" porque quanto mais rodadas sem treinamento, maior deve ser o peso
        #     # do modelo global
        #     eq2 = np.exp(eq1)
        #     eq3 = min(eq2, 1)
        #     global_model_weight = eq3
        # 4
        # max_rounds_without_fit = 3
        # alpha = 1.2
        # beta = 9
        # start_round = 5
        # # normalizar dentro de 0 e 1
        # rounds_without_fit = pow(
        #     min(rounds_without_fit + 0.0001, max_rounds_without_fit), -alpha)
        # global_model_weight = 1
        # if rounds_without_fit > 0:
        #     # o denominador faz com que a curva se prolongue com menor decaimento
        #     # Quanto mais demorada for a convergência do modelo, maior deve ser o valor do denominador
        #     eq1 = (-rounds_without_fit - (server_round - start_round) / beta)
        #     # eq2: se divide por "rounds_without_fit" porque quanto mais rodadas sem treinamento, maior deve ser o peso
        #     # do modelo global
        #     eq2 = np.exp(eq1)
        #     eq3 = min(eq2, 1)
        #     global_model_weight = eq3
        # 5
        # max_rounds_without_fit = 3
        # alpha = 1.2
        # beta = 12
        # start_round = 0
        # # normalizar dentro de 0 e 1
        # rounds_without_fit = pow(
        #     min(rounds_without_fit + 0.0001, max_rounds_without_fit), -alpha)
        # global_model_weight = 1
        # if rounds_without_fit > 0:
        #     # o denominador faz com que a curva se prolongue com menor decaimento
        #     # Quanto mais demorada for a convergência do modelo, maior deve ser o valor do denominador
        #     eq1 = (-rounds_without_fit - (server_round - start_round) / beta)
        #     # eq2: se divide por "rounds_without_fit" porque quanto mais rodadas sem treinamento, maior deve ser o peso
        #     # do modelo global
        #     eq2 = np.exp(eq1)
        #     eq3 = min(eq2, 1)
        #     global_model_weight = eq3
        # 6
        # max_rounds_without_fit = 4
        # alpha = 1.2
        # beta = 9
        # # evitar que um modelo que treinou na rodada atual não utilize parâmetros globais pois esse foi atualizado após o seu treinamento
        # delta = 0.02
        # # normalizar dentro de 0 e 1
        # fx_rounds_without_fit = pow(
        #     min(rounds_without_fit + delta, max_rounds_without_fit), -alpha)
        # # global_model_weight = 1
        # # o denominador faz com que a curva se prolongue com menor decaimento
        # # Quanto mais demorada for a convergência do modelo, maior deve ser o valor do denominador
        # eq1 = (-fx_rounds_without_fit - ((rounds_without_fit) / beta))
        # # eq2: se divide por "rounds_without_fit" porque quanto mais rodadas sem treinamento, maior deve ser o peso
        # # do modelo global
        # eq2 = np.exp(eq1)
        # eq3 = min(eq2, 1)
        # global_model_weight = eq3
        # 7
        # sigma = 5
        # mu = 10
        # eq1 = - pow((rounds_without_fit-mu), 2)/(1*sigma*sigma)
        # # eq2 = 1/(sigma*pow((2*np.pi), 1/2))
        # eq2 = 1/2
        # eq3 = eq2*np.exp(eq1)
        # global_model_weight = eq3
        # 8
        max_rounds_without_fit = 4
        alpha = 1.2
        beta = 9
        mu = -1
        # evitar que um modelo que treinou na rodada atual não utilize parâmetros globais pois esse foi atualizado após o seu treinamento
        delta = 0.02
        # normalizar dentro de 0 e 1
        updated_level = pow(
            min(rounds_without_fit + delta, max_rounds_without_fit), -alpha)
        evolutionary_level = (server_round / 50)

        eq1 = (-updated_level - evolutionary_level)
        # eq2 = 1/(sigma*pow((2*np.pi), 1/2))
        eq2 = 1
        eq3 = eq2 * np.exp(eq1)
        global_model_weight = min(eq3, 1)

        return global_model_weight

    def server_analysis(self):

        rounds = 50
        rounds_without_fit = 0
        rounds_without_fit_list = []
        start_round = 0
        x = [i for i in range(1, rounds + 1)]
        rounds_without_fit_list = rounds_without_fit_list + [rounds_without_fit] * len(x)
        y0 = [self.curve(i, rounds_without_fit, start_round) for i in x]
        rounds_without_fit = 1
        rounds_without_fit_list = rounds_without_fit_list + [rounds_without_fit] * len(x)
        y1 = [self.curve(i, rounds_without_fit, start_round) for i in x]
        rounds_without_fit = 2
        rounds_without_fit_list = rounds_without_fit_list + [rounds_without_fit] * len(x)
        y2 = [self.curve(i, rounds_without_fit, start_round) for i in x]
        rounds_without_fit = 3
        rounds_without_fit_list = rounds_without_fit_list + [rounds_without_fit] * len(x)
        y3 = [self.curve(i, rounds_without_fit, start_round) for i in x]

        x = x*4
        y = y0 + y1 + y2 + y3
        x_column = 'Evolutionary level'
        y_column = 'Global parameters weights'
        hue = 'Outdated level'
        print(len(x), len(y), len(rounds_without_fit_list))
        df = pd.DataFrame({x_column: x, y_column: y, hue: rounds_without_fit_list})
        title = ""
        line_plot(df=df,
                  base_dir=self.base_dir,
                  file_name="curve",
                  x_column=x_column,
                  y_column=y_column,
                  title=title,
                  hue=hue)

if __name__ == '__main__':

    Verify().start()