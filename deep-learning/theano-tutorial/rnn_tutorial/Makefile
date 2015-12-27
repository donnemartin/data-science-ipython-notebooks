all: instruction.pdf rnn_lstm.pdf

instruction.pdf: slides_source/instruction.tex
	cd slides_source; pdflatex --shell-escape instruction.tex
	cd slides_source; pdflatex --shell-escape instruction.tex
	cd slides_source; pdflatex --shell-escape instruction.tex
	mv slides_source/instruction.pdf .

rnn_lstm.pdf: slides_source/rnn_lstm.tex
	cd slides_source; pdflatex --shell-escape rnn_lstm.tex
	cd slides_source; pdflatex --shell-escape rnn_lstm.tex
	cd slides_source; pdflatex --shell-escape rnn_lstm.tex
	mv slides_source/rnn_lstm.pdf .
