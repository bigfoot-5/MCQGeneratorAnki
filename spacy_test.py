from lemminflect import getLemma, getInflection, getAllInflections
x = getAllInflections('teach', upos="VERB")
print('\n')
print(x)

x = getLemma('watch', upos='VERB')
print('\n')
print(x)

x = getInflection('watch', tag='VBD')
print('\n')
print(x)
