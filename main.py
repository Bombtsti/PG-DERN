import json
import os
print('pid:', os.getpid())

from time import time
from parser import get_args
from model.models import DERN, Meta_Trainer
from model.utils import count_model_params

def main():
    root_dir = '.'
    args = get_args(root_dir)

    model = DERN(args)
    count_model_params(model)
    model = model.to(args.device)
    trainer = Meta_Trainer(args, model)

    t1=time()
    print('Initial Evaluation')
    best_avg_auc=0
    best = 0
    for epoch in range(1, args.epochs + 1):
        print('----------------- Epoch:', epoch,' -----------------')
        res = trainer.train_step(epoch)
        f_tasks = res[1]

        if epoch % args.eval_steps == 0 or epoch==1 or epoch==args.epochs:
            print('Evaluation on epoch',epoch)
            best_avg_auc,ts = trainer.test_step(epoch,f_tasks)

        # if epoch % args.save_steps == 0:
        #     trainer.save_model()
        if best_avg_auc>best:
            trainer.save_model()
            # print(ts)
            # with open('ts.json', 'w') as json_file:
            #     for key, tensors in ts.items():
            #         json.dump({key: tensor.tolist() for key, tensor in tensors.items()}, json_file)
            best=best_avg_auc
        print('Time cost (min):', round((time()-t1)/60,3))
        t1=time()

    print('Train done.')
    print('Best Avg AUC:',best_avg_auc)

    trainer.conclude()

    if args.save_logs:
        trainer.save_result_log()

if __name__ == "__main__":
    main()
