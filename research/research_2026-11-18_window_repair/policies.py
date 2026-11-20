"""Length-class separation and a bounded conservative alternative for CPU replay."""
from array import array


def window_floor(length, window):
    units=length//window
    return 0 if units<2 else window*(1<<(units.bit_length()-1))


def removable_floor(prefix,length,chunk,floor,window):
    stop=max(0,floor-window)
    return max(0,(stop-prefix)//chunk)*chunk


class RepairLedger:
    """Monotone potential-omission by prefix bucket; collisions only add repair work."""
    def __init__(self,window,key_length=64,buckets=4096):
        self.window,self.key_length=window,key_length
        self.high=array("Q",[0])*buckets

    def bucket(self,ids):
        return hash(tuple(ids[:self.key_length]))%len(self.high)

    def mark(self,ids,start,end):
        assert start>=self.key_length and end<=len(ids)-self.window
        b=self.bucket(ids)
        self.high[b]=max(self.high[b],end)

    def boundary(self,ids,length):
        if length<=self.key_length: return None
        first=max(self.key_length,length-self.window)
        if first<min(length,self.high[self.bucket(ids)]):
            return first
        return None

    def limit(self,ids,length,ordinary):
        cutoff=self.boundary(ids,length)
        return ordinary if cutoff is None else min(ordinary,cutoff)

    def publish(self,ids,length):
        return self.boundary(ids,length) is None


def repair_omissions(ledger,ids,prefix,end,chunk):
    result=[]
    while prefix<end:
        stop=min(end,prefix+chunk)
        if prefix>=ledger.key_length and stop<=end-ledger.window:
            ledger.mark(ids,prefix,stop)
            result.append((prefix,stop))
        prefix=stop
    return result

