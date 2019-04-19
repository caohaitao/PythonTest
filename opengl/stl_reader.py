import copy

class triangle:
    def __init__(self,n,ps):
        self.normal = n
        self.pnts = ps

def convert_line_to_pnt(line):
    strs = line.split(' ')
    return [float(strs[-3]), float(strs[-2]), float(strs[-1])]

def read_one_file(file_path):
    triangles = []
    IsReading = False
    n = []
    ps = []
    err_nums = 0
    for line in open(file_path):
        line = line.replace('\n','')
        if line.find('facet normal') != -1:
            n = convert_line_to_pnt(line)
            IsReading = True
        elif line.find('vertex') != -1:
            if IsReading and len(ps)<3:
                ps.append(convert_line_to_pnt(line))
        elif line.find('endfacet') != -1:
            if IsReading == False:
                continue
            IsReading = False
            if len(ps)==3:
                triangles.append(triangle(copy.deepcopy(n),copy.deepcopy(ps)))
                ps.clear()
        elif line.find("outer loop")!=-1 or line.find("endloop")!=-1:
            continue
        elif line.find("endsolid") != -1:
            break
        else:
            err_nums+=1
            if err_nums>100:
                return None

    return triangles



if __name__=='__main__':
    read_one_file(r"D:\code\MyCodes\CHT3D\modelmatch\stlModels\mytest\rect0.stl")