package src;

import weka.core.Instances;
import weka.classifiers.functions.LinearRegression;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class linearRegression {
    public static LinearRegression main(Instances dataset) throws Exception {
        /*
        String dataSource = args[0];

        Instances dataset = loadData(dataSource);
        if (dataset == null) {
            System.out.println("Mesedez, egiaztatu helbidea ondo sartu duzula.");
            return null;
        }
        */
        dataset = preprocessData(dataset);
        dataset.setClassIndex(dataset.numAttributes() - 1);

        LinearRegression model = buildModel(dataset);
        if (model != null)
            return model;
        return null;
    }

    private static Instances preprocessData(Instances dataset) {
        try {
            // Check if there are string attributes
            boolean hasStringAttributes = false;
            for (int i = 0; i < dataset.numAttributes(); i++) {
                if (dataset.attribute(i).isString()) {
                    hasStringAttributes = true;
                    break;
                }
            }

            if (hasStringAttributes) {
                // Apply StringToWordVector filter
                StringToWordVector stringToWordVector = new StringToWordVector();
                stringToWordVector.setInputFormat(dataset);
                dataset = Filter.useFilter(dataset, stringToWordVector);
            }

            // Binary attributes are already numeric (0 or 1), so no additional processing is needed
            return dataset;
        } catch (Exception e) {
            System.out.println("ERROREA: Ezin izan da datuak aurreprozesatu.");
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