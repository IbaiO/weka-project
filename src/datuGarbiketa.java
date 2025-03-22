package src;

import weka.core.Instances;
import weka.core.Instance;
import java.util.regex.Pattern;

@SuppressWarnings("all")
public class datuGarbiketa {

    private static datuGarbiketa nireDG = null;

    public static datuGarbiketa getDatuGarbiketa() {
        if (nireDG == null) {
            nireDG = new datuGarbiketa();
        } return nireDG;
    }

    public Instances garbitu(Instances data) {
        Pattern hashtagPattern = Pattern.compile("#\\w+");
        Pattern punctuationPattern = Pattern.compile("\\p{Punct}");

        for (int i = 0; i < data.numInstances(); i++) {
            Instance instance = data.instance(i);
            for (int j = 0; j < instance.numAttributes(); j++) {
                if (instance.attribute(j).isString()) {
                    String text = instance.stringValue(j);
                    text = text.replaceAll("\"", ""); // Komatxoak kendu
                    text = text.toLowerCase(); // Letra xehetan bihurtu
                    text = hashtagPattern.matcher(text).replaceAll(""); // Hashtagak kendu
                    text = punctuationPattern.matcher(text).replaceAll(""); // Puntuazioak kendu
                    instance.setValue(j, text);
                }
            }
        }
        return data;
    }
}
