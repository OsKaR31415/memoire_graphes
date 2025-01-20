"""
procedure ORDREK(x, y, v)
    if y > n−k and deg[x] < k and n−y < k−deg[x] then
        return
    end if
    if x <= n−k and deg[x]==k then
        for i = y + 1 to n do
            if n−x−1 < k−deg[i] then
                return
            end if
        end for
    end if
    while x < n and deg[x]==k do
        x ∶= x + 1
    end while
    if v <= y then
        v ∶= y + 1
    end if
    if x == v then
        return
    end if
    if KATEST() == 0 then
        return
    end if
    if x == n and deg[x] == k then
        OUTPUT()
    end if
    y∶=x
    while y < n do
        y ∶= y + 1
        if deg[y] < k then
            INSERT(x, y)
            ORDREK(x, y, v)
            DELETE(x, y)
        end if
    end while
    return
end procedure
"""


def ORDREK(x, y, v):
    if y > n−k and deg[x] < k and n−y < k−deg[x]:
        return
    if x <= n−k and deg[x] == k:
        for i in range(y+1, n+1):
            if n−x−1 < k−deg[i]:
                return
    while x < n and deg[x] == k:
        x = x + 1
    if v <= y:
        v = y + 1
    if x == v:
        return
    if KATEST() == 0:
        return
    if x == n and deg[x] == k:
        OUTPUT()
    y = x
    while y < n:
        y = y + 1
        if deg[y] < k:
            INSERT(x, y)
            ORDREK(x, y, v)
            DELETE(x, y)
    return


