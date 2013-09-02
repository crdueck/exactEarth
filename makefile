PANDOC = pandoc
SOURCE = functional_lunch_and_learn

slidy:
	${PANDOC} -t $@ -s ${SOURCE}.md -o ${SOURCE}.html

beamer:
	${PANDOC} -t $@ -s ${SOURCE}.md -o ${SOURCE}.pdf

clean:
	rm -f *.aux *.log *.pdf *.html
