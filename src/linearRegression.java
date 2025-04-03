package src;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.classifiers.functions.LinearRegression;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.util.ArrayList;

public class linearRegression {
    public static LinearRegression main(Instances dataset) throws Exception {
        // Klasearen indizea ezarri.
        if (dataset.classIndex() == -1) {
            dataset.setClassIndex(dataset.numAttributes() - 1); // Set the last attribute as the class
        }

        // Datuak prozesatu.
        dataset = preprocessData(dataset);
        if (dataset == null) {
            System.out.println("ERROREA: Ezin izan dira datuak prozesatu.");
            return null;
        }

        // Modeloa eraiki.
        LinearRegression model = buildModel(dataset);
        if (model == null) {
            System.out.println("ERROREA: Ezin izan da modeloa eraiki.");
            return null;
        }

        return model;
    }

    private static Instances preprocessData(Instances dataset) {
        try {
            // Ensure the class index is set before preprocessing
            if (dataset.classIndex() == -1) {
                dataset.setClassIndex(dataset.numAttributes() - 1); // Set the last attribute as the class
            }

            // Move the class attribute to the first position
            if (dataset.classIndex() != 0) {
                ArrayList<Attribute> attributes = new ArrayList<>();
                attributes.add(dataset.attribute(dataset.classIndex())); // Add the class attribute first
                for (int i = 0; i < dataset.numAttributes(); i++) {
                    if (i != dataset.classIndex()) {
                        attributes.add(dataset.attribute(i));
                    }
                }

                Instances newDataset = new Instances(dataset.relationName(), attributes, dataset.numInstances());
                newDataset.setClassIndex(0); // Set the class index to the first position

                // Update instances to match the new attribute order
                for (int i = 0; i < dataset.numInstances(); i++) {
                    DenseInstance newInstance = new DenseInstance(newDataset.numAttributes());
                    newInstance.setDataset(newDataset);

                    // Set the class value first
                    newInstance.setValue(0, dataset.instance(i).classValue());
                    // Set the remaining attribute values
                    int newIndex = 1;
                    for (int j = 0; j < dataset.numAttributes(); j++) {
                        if (j != dataset.classIndex()) {
                            newInstance.setValue(newIndex++, dataset.instance(i).value(j));
                        }
                    }

                    newDataset.add(newInstance);
                }

                dataset = newDataset;
            }

            // StringToWordVector iragazkia aplikatu, behar izanez gero.
            boolean hasStringAttributes = false;
            for (int i = 0; i < dataset.numAttributes(); i++) {
                if (dataset.attribute(i).isString()) {
                    hasStringAttributes = true;
                    break;
                }
            }

            if (hasStringAttributes) {
                StringToWordVector stringToWordVector = new StringToWordVector();
                stringToWordVector.setInputFormat(dataset);
                dataset = Filter.useFilter(dataset, stringToWordVector);

                // Reset the class index after applying the filter
                dataset.setClassIndex(0); // Ensure the class index remains at the first position
            }

            // Atributu bitar nominalak zenbakizkotan bihurtu
            if (dataset.classIndex() != -1) {
                int classIndex = dataset.classIndex();
                if (dataset.attribute(classIndex).isNominal() && dataset.attribute(classIndex).numValues() == 2) {
                    // Atributu zerrenda bat sortu, DataSet berriarentzat
                    ArrayList<Attribute> attributes = new ArrayList<>();
                    for (int i = 0; i < dataset.numAttributes(); i++) {
                        if (i == classIndex) {
                            // Reemplazar el atributo de clase nominal con un atributo numérico
                            attributes.add(new Attribute("class"));
                        } else {
                            attributes.add(dataset.attribute(i));
                        }
                    }

                    // DataSet berria sortu Pos/Neg klasea zenbaki gisa jarrita
                    // Pos: 1, Neg: 0
                    Instances newDataset = new Instances(dataset.relationName(), attributes, dataset.numInstances());
                    newDataset.setClassIndex(classIndex);
                    // Instantziak DataSet berrira gehitu
                    for (int i = 0; i < dataset.numInstances(); i++) {
                        DenseInstance newInstance = new DenseInstance(newDataset.numAttributes());
                        newInstance.setDataset(newDataset);

                        for (int j = 0; j < dataset.numAttributes(); j++) {
                            if (j == classIndex) {
                                String classValue = dataset.instance(i).stringValue(classIndex);
                                if (classValue.equalsIgnoreCase("pos")) {
                                    newInstance.setValue(classIndex, 1);
                                } else if (classValue.equalsIgnoreCase("neg")) {
                                    newInstance.setValue(classIndex, 0);
                                } else if (classValue.equals("?")) {
                                    newInstance.setMissing(classIndex);
                                } else {
                                    newInstance.setValue(classIndex, -1);
                                }
                            } else {
                                newInstance.setValue(j, dataset.instance(i).value(j));
                            }
                        }

                        newDataset.add(newInstance);
                    }
                    dataset = newDataset;
                }
            }

            return dataset;
        } catch (Exception e) {
            System.out.println("ERROREA: Ezin izan dira datuak prozesatu.");
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