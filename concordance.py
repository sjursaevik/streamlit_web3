import spacy
import streamlit as st
import spacy_streamlit as sst
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud


st.set_page_config(layout='wide')
nlp=spacy.load('./model-last')
nlp2=spacy.load('./model-best')


#morphologizer=Morphologizer(vocab=nlp2.vocab, model='nb_core_news_lg', name='morphologizer')
#nlp_nb=spacy.load('nb_core_news_lg', vocab=nlp2.vocab)
#mconfig={}

#nlp2.add_pipe('morphologizer',source=spacy.load('nb_core_news_lg', vocab=nlp2.vocab), before='ner')


#text='''Det solfylte sommerbildet med brudefølget i båtene, stavkirken på odden og vestlands- naturen med fjord og fjell er et typisk uttrykk for nasjonalromantikkens opplevelse av norsk natur og folkeliv. Kunstnerne spilte en vesentlig rolle når det gjaldt å definere en nasjonal egenart etter at Norge hadde fått sin grunnlov i 1814. Dette motivet, som så sterkt uttrykker 1800-tallets skjønnhetsidealer, har vært dyrket som et ”ikon” av generasjoner av nordmenn. Maleriet har vært overført til teaterscenen både som levende tablå og ballett, og motivet er blitt ledsaget av dikt og musikk.

#Gjennom grafiske reproduksjoner fikk Brudeferd i Hardanger stor utbredelse, og på grunn av motivets spesielle popularitet har kunstnerne utført maleriet i flere versjoner. Adolph Tidemand var den første norske kunstner som slo seg ned i Düsseldorf. Sin ambisjon om å bli historiemaler oppga han for å bli folkelivsskildrer. Men Tidemand gir en ny verdighet til bøndene, og dikteren Bjørnstjerne Bjørnson skal ha sagt at uten Tidemands malerier hadde han ikke kunnet skrive sine bondefortellinger.

#Landskapsmaler Hans Gude, som var drøye ti år yngre enn Tidemand, presenterer her som 23-åring en storslagen skildring av norsk natur. Selv om det ikke dreier seg om en direkte gjengivelse av et bestemt landskap, er komposisjonen satt sammen av nøyaktige naturobservasjoner fra forskjellige steder i hjemlandet. Tidemand og Gude har utført flere malerier sammen, der alle motivene viser folk som er ute i båt.'''

def concordance(doc, size):
    conc=[]
    for ent in doc.ents:
        try:
            indexb=ent.start_char - size
        except: 
            indexb=0
        try:
            indexe=ent.end_char + size
        except: 
            indexe=len(doc.text)
        conc.append([ent.label_, ent, doc.text[(indexb):(indexe)]])
    return conc
text = st.text_area('sett in tekst her: ', value='''Det solfylte sommerbildet med brudefølget i båtene, stavkirken på odden og vestlands- naturen med fjord og fjell er et typisk uttrykk for nasjonalromantikkens opplevelse av norsk natur og folkeliv. Kunstnerne spilte en vesentlig rolle når det gjaldt å definere en nasjonal egenart etter at Norge hadde fått sin grunnlov i 1814. Dette motivet, som så sterkt uttrykker 1800-tallets skjønnhetsidealer, har vært dyrket som et ”ikon” av generasjoner av nordmenn. Maleriet har vært overført til teaterscenen både som levende tablå og ballett, og motivet er blitt ledsaget av dikt og musikk.

#Gjennom grafiske reproduksjoner fikk Brudeferd i Hardanger stor utbredelse, og på grunn av motivets spesielle popularitet har kunstnerne utført maleriet i flere versjoner. Adolph Tidemand var den første norske kunstner som slo seg ned i Düsseldorf. Sin ambisjon om å bli historiemaler oppga han for å bli folkelivsskildrer. Men Tidemand gir en ny verdighet til bøndene, og dikteren Bjørnstjerne Bjørnson skal ha sagt at uten Tidemands malerier hadde han ikke kunnet skrive sine bondefortellinger.

#Landskapsmaler Hans Gude, som var drøye ti år yngre enn Tidemand, presenterer her som 23-åring en storslagen skildring av norsk natur. Selv om det ikke dreier seg om en direkte gjengivelse av et bestemt landskap, er komposisjonen satt sammen av nøyaktige naturobservasjoner fra forskjellige steder i hjemlandet. Tidemand og Gude har utført flere malerier sammen, der alle motivene viser folk som er ute i båt.''')
#print(text)


doc=nlp(text)




doc2=nlp2(text)
cc=concordance(doc, 50)
df=pd.DataFrame(cc, columns=['Label', 'Entity', 'Context']).sort_values('Label')
df = df.astype(str)
df=df.style
cc2=concordance(doc2, 50)
df2=pd.DataFrame(cc2, columns=['Label', 'Entity', 'Context']).sort_values('Label')
df2 = df2.astype(str)
df2=df2.style

nouns=[]
#print(nlp2.pipe_names)
for tok in doc2:
    dep=tok.is_stop
    #print(tok, dep)
    word=tok.text
    word=word.replace("'", "")
    #print (word)
    if dep==False:
        nouns.append(word)

#print(type(nouns), nouns)
#nouns=sorted(nouns)
#print(type(nouns.sort()), nouns.sort())

nouns=str(nouns)
nouns=nouns.replace("'", "")
nouns=nouns.strip("[").strip(']')
#print(nouns)
st.write('**Entiter i kontekst med labels, beste modell**')      
st.dataframe(df2, use_container_width=True)
st.write('**Entiter i kontekst med labels, sist trente modell**')      
st.dataframe(df, use_container_width=True)
st.write('**Hele teksten med labels, sist trente modell**')
sst.visualize_ner(doc, labels=nlp.get_pipe("ner").labels, key='s1', show_table=False)
st.write('**Hele teksten med labels, beste modell**')
sst.visualize_ner(doc2, labels=nlp2.get_pipe("ner").labels, key='s2', show_table=False)
st.write('''**Ordsky for teksten**''') 
wordcloud= WordCloud(collocations=False, min_word_length=3, width=800, height=800, background_color='white').generate(str(nouns))
fig, ax = plt.subplots(figsize = (12, 8))
ax.imshow(wordcloud)
plt.axis("off")
st.pyplot(fig)

doc.count_by()
pdata={'person':[], 'frequency':[]}
persons=pd.DataFrame(data=pdata)

st.write(persons)