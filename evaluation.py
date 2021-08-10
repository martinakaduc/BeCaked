from becaked import BeCakedModel
from data_utils import DataLoader
import tensorflow as tf
from utils import *
import argparse

float_formatter = "{:.6f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--level', help='0: world; 1: countries; 2: both', type=int, default=0)
    parser.add_argument('--day_lag', help='The number of day lag.', type=int, default=10)
    parser.add_argument('--step', help='The number of forecasting step.', type=int, default=1)
    parser.add_argument('--start_date', help='The start day from which to make prediction.', type=int, default=161)
    parser.add_argument('--end_date', help='The end date of prediction.', type=int, default=192)
    parser.add_argument('--start_train_date', help='The start date of trainning set.', type=int, default=0)
    parser.add_argument('--end_train_date', help='The end date of trainning set.', type=int, default=53)
    parser.add_argument('--infer_date', help='The end date of inference.', type=int, default=10)
    parser.add_argument('--run_comparison', help='Wheather to compare model.', type=bool, default=False)
    parser.add_argument('--plot_prediction', help='Wheather to plot prediction.', type=bool, default=False)
    parser.add_argument('--plot_param', help='Wheather to plot parameters.', type=bool, default=False)
    parser.add_argument('--image_folder', help='Where to save plotted pictures.', type=str, default="./images")
    parser.add_argument('--ward', help='Ward name', type=str, default="HCM")
    parser.add_argument('--cuda', help='Enable cuda', type=int, default=0)
    parser.add_argument('--limit_gpu', help='Limit GPU', type=int, default=1)
    parser.add_argument('--img_note', help='figname', type=str, default="")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        

    if not os.path.exists(args.image_folder):
        os.makedirs(args.image_folder)

    data_loader = DataLoader(ward_name=args.ward)

    if args.level == 0 or args.level == 2:
        print("===================== WORLD =====================")
        becaked_model = BeCakedModel(population=data_loader.N, day_lag=args.day_lag)
        data = data_loader.get_data_world_series()

        if not os.path.exists("models/%s_%d.h5" % ("world", args.day_lag)):
            print("Model does not exist. Trying to train...")
            becaked_model.train(data[0][args.start_train_date:args.end_train_date],
                                data[1][args.start_train_date:args.end_train_date],
                                data[2][args.start_train_date:args.end_train_date],
                                data[3][args.start_train_date:args.end_train_date],
                                data[4][args.start_train_date:args.end_train_date],
                                data[5][args.start_train_date:args.end_train_date],
                                epochs=1000)

        if args.run_comparison:
            get_all_compare(data, becaked_model, args.start_date, args.end_date, step=args.step, day_lag=args.day_lag)

        if args.plot_prediction or args.plot_param:
            predict_data_0, list_param_byu_0 = get_predict_result_1(becaked_model, data,
                                                                    args.start_date, args.end_date-5,
                                                                    end=args.end_date-5, day_lag=args.day_lag,
                                                                    return_param=True)

            predict_data_1, list_param_byu_1 = get_predict_by_step(becaked_model, data,
                                                                    args.end_date, args.end_date-5,
                                                                    end=args.end_date + args.infer_date,
                                                                    day_lag=args.day_lag, return_param=True)

            predict_data = np.append(predict_data_0,predict_data_1[:,args.end_date-5:],axis=1)
            list_param_byu = np.append(list_param_byu_0, list_param_byu_1[args.end_date-5:], axis=0)

            # print(predict_data)
            # print(list_param_byu)
            if args.plot_prediction:
                plot(data, predict_data, list_param_byu, args.start_date-args.day_lag, args.end_date, country=args.ward, idx=args.img_note)
            if args.plot_param:
                plotParam(list_param_byu, args.start_date-args.day_lag, args.end_date, country="world")

    ###################### COUNTRY LEVEL #############################
    if args.level == 1 or args.level == 2:
        confirmed_countries, recovered_countries, deceased_countries = data_loader.get_data_countries_series()
        countries = ["Australia", "Italy", "Russia", "Spain", "US", "United Kingdom"]
        countries_population = [25e6, 60e6, 144.5e6, 46.5e6, 328e6, 66.5e6]

        for i, country in enumerate(countries):
            becaked_model = BeCakedModel(population=countries_population[i], day_lag=args.day_lag)

            print("===================== %s =====================" % country)
            data = np.array([confirmed_countries[country], recovered_countries[country], deceased_countries[country]], dtype=np.float64)

            if not os.path.exists("models/%s_%d.h5" % (country, args.day_lag)):
                print("Model does not exist. Trying to train...")
                becaked_model.train(data[0][:args.start_date], data[1][:args.start_date], data[2][:args.start_date], epochs=10000, name=country)

            becaked_model.load_weights("models/%s_%d.h5" % (country, args.day_lag))

            if args.run_comparison:
                get_all_compare(data, becaked_model, args.start_date, args.end_date, step=args.step, day_lag=args.day_lag)

            if args.plot_prediction or args.plot_param:
                for i in range (args.end_date - args.start_date):
                    predict_data, list_param_byu = get_predict_by_step(becaked_model, data, args.start_date, args.start_date + i, end=args.end_date, day_lag=args.day_lag, return_param=True)

                    if args.plot_prediction:
                        plot(data, predict_data, args.start_date-args.day_lag, args.end_date, country=country, idx=str(i))
                    if args.plot_param:
                        plotParam(list_param_byu, args.start_date-args.day_lag, args.end_date, country=country, idx=str(i))
