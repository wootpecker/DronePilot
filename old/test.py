import csv


def save_to_csv(t,position_estimate):
    with open('./data/test.tsv', 'a+', newline='') as f:
        tsv_w = csv.writer(f, delimiter='\t')
        tsv_w.writerow([int(t), format(position_estimate[0], '.10f'), format(position_estimate[1], '.10f'), format(position_estimate[2], '.10f')])


time=[]
position_estimate=[]
for x in range(10, 110):
    time.append(x)
    position_estimate.append([x, 5, 5])
    save_to_csv(x,[x, 5, 5])
    

for x in range(time[0]):
    save_to_csv(time[x],position_estimate[x])


