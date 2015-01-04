import sys
from xml.etree import ElementTree

if __name__ == '__main__':
  input_file = sys.argv[1]
  tree = ElementTree.parse(input_file)
  topics = tree.findall("topic")
  print len(topics)
  for tp in topics:
    tid = tp.get('id')
    title = tp.find("title").text.strip()
    narrative = tp.find("narrative").text.strip()
    outstr = tid + '####' + title + '####' + narrative
    print outstr
