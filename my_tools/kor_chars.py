import sys

out = ''
f = open(sys.argv[1], 'r') 
lines = f.readlines()
for word in lines:
  out = out + word.strip()
# lines = list(map(lambda s: s.strip(), lines))
print(out)