import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.BestFirst;
import weka.core.Utils;

public class baterazintasunakEtaIragarpena {
    public static void main(String[] args) throws Exception {
        if(args.length < 2) {
            System.out.println("Ez duzu sartu beharreko parametro guztiak");
            System.exit(1);
        }
        String trainArff = args[0];
        String testBlindArff = args[1];
        // Load the training set
        DataSource trainSource = new DataSource(trainArff);
        Instances train = trainSource.getDataSet();
        if (train.classIndex() == -1) {
            train.setClassIndex(train.numAttributes() - 1);
        }

        // Load the test set
        DataSource testSource = new DataSource(testBlindArff);
        Instances test = testSource.getDataSet();
        if (test.classIndex() == -1) {
            test.setClassIndex(test.numAttributes() - 1);
        }

        // Configure the AttributeSelectedClassifier
        AttributeSelectedClassifier classifier = new AttributeSelectedClassifier();
        classifier.setClassifier(new NaiveBayes());
        classifier.setEvaluator(new CfsSubsetEval());
        classifier.setSearch(new BestFirst());

        // Train the classifier
        classifier.buildClassifier(train);

        // Make predictions on the test set
        for (int i = 0; i < test.numInstances(); i++) {
            double label = classifier.classifyInstance(test.instance(i));
            test.instance(i).setClassValue(label);
            System.out.println("Instance " + (i + 1) + ": Predicted label = " + test.classAttribute().value((int) label));
        }
    }
}
