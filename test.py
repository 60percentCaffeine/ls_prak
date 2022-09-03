from app import preproc, print_sentences
f=open('weller_convey.txt','r')
doc=preproc(f.read())
doc.make_dictionary()
doc.cwi()
print_sentences(doc.sentences[0:2])
