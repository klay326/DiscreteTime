import discretetime as dt
x = dt.Signal([-1, 2, 3, -2], -2, 1)
y = dt.Signal([2, 0, -3, 1], 0, 3)

x, y = dt.Signal.matchSignals(x,y)
x.plot()
#y.plot()
#w = x + y
#w.plot("w[n]")

#z = x.flip()
#z.plot("z[n]")

#k = x.shift(10)
#k.plot("k[n]")

j = x.expand(3)
j.plot("j[n]")
