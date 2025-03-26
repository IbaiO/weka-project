package src;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.functions.LinearRegression;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.BestFirst;

public class linearRegression {
    public static void main(String[] args) {
        String dataSource = "/home/ibai/GitHub/weka-project/probaData/toyStringExample_NonSparseBoW.arff";

        Instances dataset = loadData(dataSource);
        if (dataset == null) {
            System.out.println("Mesedez, egiaztatu helbidea ondo sartu duzula.");
            return;
        }

        dataset.setClassIndex(dataset.numAttributes() - 1);

        // Perform feature selection
        Instances selectedDataset = selectAttributes(dataset);
        if (selectedDataset == null) {
            System.out.println("Errorea: Ezin izan da atributuen hautaketa burutu.");
            return;
        }

        LinearRegression model = buildModel(selectedDataset);
        if (model != null) {
            System.out.println("LR FORMULA : " + model);
            predictPrice(model, selectedDataset);
        }
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

    private static Instances selectAttributes(Instances dataset) {
        try {
            AttributeSelection attributeSelection = new AttributeSelection();
            CfsSubsetEval evaluator = new CfsSubsetEval();
            BestFirst search = new BestFirst();
            attributeSelection.setEvaluator(evaluator);
            attributeSelection.setSearch(search);
            attributeSelection.SelectAttributes(dataset);
            return attributeSelection.reduceDimensionality(dataset);
        } catch (Exception e) {
            System.out.println("ERROREA: Ezin izan da atributuen hautaketa burutu.");
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

    private static void predictPrice(LinearRegression model, Instances dataset) {
        try {
            Instance myHouse = dataset.lastInstance();
            double price = model.classifyInstance(myHouse);
            System.out.println("-------------------------");
            System.out.println("PRECTING THE PRICE: " + price);
        } catch (Exception e) {
            System.out.println("ERROREA: Ezin izan da prezioa aurreikusi.");
            e.printStackTrace();
        }
    }
}