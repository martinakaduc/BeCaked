temp = """'{}':
            {{
                'I': {{'New': {}, 'Total': {}}},
                'D': {{'New': {}, 'Total': {}}}, 
                'R': {{'New': {}, 'Total': {}}}
            }},
            
        """

import random
districts = ['BINH CHANH', 'BINH TAN', 'BINH THANH', 'CAN GIO', 'CU CHI', 'GO VAP', 'HCM', 'HOC MON', 'NHA BE', 'PHU NHUAN'] + [f'QUAN {i}' for i in [1, 3, 4, 5, 6, 7, 8, 10, 11, 12]] + ['TAN BINH', 'TAN PHU', 'THU DUC']
out = ''
for d in districts:
    x = [d]
    for i in range(6):
        x.append(random.randint(0,1000))
    out += temp.format(*x)

print(out)
