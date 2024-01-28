import selector, time

c = int(1e8)

now = time.time()
selector.run(c)
end = time.time()
print(f'运行时间为：{end - now}秒')
