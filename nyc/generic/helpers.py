def eval_list(l, objtype=int):
    if "," in l:
        return eval(l)
    else:
        # print([(x.replace("[", "").replace("]", "")) for x in l.split()])
        return [objtype(x.replace("[", "").replace("]", "")) for x in l.split() if len(x.replace("[", "").replace("]", "")) > 0]
