import re
content = open('prompts/task1/generator_combined.txt', encoding='utf-8').read()
titles = re.findall(r'Title .+ "(.+?)"', content)
print(f'Chars: {len(content)}')
print(f'Example blocks: {content.count(chr(10) + "Example ")}')
print(f'Unique titles: {len(set(titles))}/{len(titles)}')
for t in sorted(set(titles)):
    print(f'  [{titles.count(t)}x] {t[:60]}')
