package src;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.functions.LinearRegression;

public class linearRegression {
    public static LinearRegression linearRegressionSortu(String[] args) {
        String dataSource = "/home/ibai/GitHub/weka-project/probaData/toyStringExample_NonSparseBoW.arff";

        Instances dataset = loadData(dataSource);
        if (dataset == null) {
            System.out.println("Mesedez, egiaztatu helbidea ondo sartu duzula.");
            return null;
        }

        dataset.setClassIndex(dataset.numAttributes() - 1);

        LinearRegression model = buildModel(dataset);
        if (model != null)
            return model;
        return null;
    }

    private static Instances loadData(String dataSource) {
        try {
            DataSource source = new DataSource(dataSource);
            return source.getDataSet();
        } catch (Exception e) {
            System.out.println("ERROREA: Ezin izan da datu multzoa kargatu.");
            e.printStackTrace();
            return null;
        }
    }

    private static LinearRegression buildModel(Instances dataset) {
        try {
            LinearRegression model = new LinearRegression();
            model.buildClassifier(dataset);
            return model;
        } catch (Exception e) {
            System.out.println("ERROREA: Ezin izan da modeloa eraiki.");
            e.printStackTrace();
            return null;
        }
    }
}