#!/usr/bin/python
#-*- coding: utf-8 -*-
# encoding: utf-8

import matplotlib.pyplot as plt
import sys

def get_data(entrada1, entrada2, entrada3):

	files = list()
	files.append(open(entrada1))
	files.append(open(entrada2))
	files.append(open(entrada3))

	error = [[],[],[],[]]

	for i in range(100):

		for j in range(3):

			fields = files[j].readline().strip().split('\t')
			error[j].append(float(fields[2].strip().split('=')[1]))

	return error


def plot_time_series(errors, image_file):
	
	epochs = range(100)

	fig, ax = plt.subplots()

	ax.plot(epochs,errors[0],':', linewidth = 2, label=u'25 neurônios', color='green')
	ax.plot(epochs,errors[1],'--', linewidth = 2, label=u'50 neurônios', color='red')
	ax.plot(epochs,errors[2],'-', linewidth = 2, label=u'100 neurônios', color='blue')


	#evita que o matplolib adicione offset no eixo x
	ax.ticklabel_format(useOffset=False, style='plain')

	ax.set_ylim([3.25,3.35])
	ax.xaxis.grid()
	#ax.grid(True)

	ax.legend()

	plt.ylabel(u"Erro médio")
	plt.xlabel(u"Épocas")

	plt.show()
	#plt.savefig(image_file)

if __name__ == '__main__':
	errors = get_data(sys.argv[1], sys.argv[2], sys.argv[3])
	plot_time_series(errors, 'hl_comp.png')