/*
	Thomas Scanlon, Ayden Wilson
	
	NNP: An implementation of the LD-kNN algorithm, as defined in 
	Nearest Neighbor Method Based on Local Distribution for Classification,
	link.springer.com/chapter/10.1007/978-3-319-18038-0_19
	
	NNP extends Weka's IBk class for Instance based learning and uses
	an internal Naivebayes classifier to classify instances.
	It includes a new option, kcp, that defines the size of the local 
	distribution.
*/

package weka.classifiers.lazy;

import java.util.Collections;
import java.util.Enumeration;
import java.util.Vector;
import java.util.ArrayList;
import java.util.Arrays;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.UpdateableClassifier;
import weka.classifiers.rules.ZeroR;
import weka.core.AdditionalMeasureProducer;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.RevisionUtils;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.core.neighboursearch.LinearNNSearch;
import weka.core.neighboursearch.NearestNeighbourSearch;
import weka.classifiers.lazy.IBk;
import weka.classifiers.bayes.NaiveBayes;





public class NNP extends IBk {
	
	// value for neighbourhood size
	int kcp = 1;
	
		// MAIN
	public static void main(String[] args){
		runClassifier(new NNP(), args);
	}
	
	public void setKCP(int d){
		kcp=d;
	}
	
	public int getKCP(){
		return kcp;
	}
	
	// Options
	@Override
	public void setOptions(String[] options) throws Exception {
		
    String k = Utils.getOption("kcp", options);
    if (k.length() != 0) {
      setKCP(Integer.parseInt(k));
    } else {
      setKCP(1);
    }

    super.setOptions(options);
    
    Utils.checkForRemainingOptions(options);	
		
	}
	
	@Override
	public String[] getOptions(){
		
    Vector<String> options = new Vector<String>();
    options.add("-kcp"); options.add("" + getKCP());
	
    Collections.addAll(options, super.getOptions());
    
    return options.toArray(new String[0]);
		
	}
	
	@Override
	public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();
    result.disableAll();

    // attributes
    result.enable(Capability.NOMINAL_ATTRIBUTES);
    result.enable(Capability.NUMERIC_ATTRIBUTES);
    result.enable(Capability.MISSING_VALUES);

    // class
    result.enable(Capability.NOMINAL_CLASS);

    // instances
    result.setMinimumNumberInstances(0);
    
    return result;
  }
	
	
	
	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		
		int cls_num = instance.numClasses();
		
	    if (m_Train.numInstances() == 0) {
			//throw new Exception("No training instances!");
			return m_defaultModel.distributionForInstance(instance);
		}
		if ((m_WindowSize > 0) && (m_Train.numInstances() > m_WindowSize)) {
			m_kNNValid = false;
			boolean deletedInstance=false;
			while (m_Train.numInstances() > m_WindowSize) {
			m_Train.delete(0);
		}
		//rebuild datastructure KDTree currently can't delete
		if(deletedInstance==true)
			m_NNSearch.setInstances(m_Train);
		}

		// Select k by cross validation
		if (!m_kNNValid && (m_CrossValidate) && (m_kNNUpper >= 1)) {
			crossValidate();
		}

		m_NNSearch.addInstanceInfo(instance);

		// Get nearest neighbours
		Instances neighbours = m_NNSearch.kNearestNeighbours(instance, m_kNN);
		
		Instances local = new Instances(neighbours, 0);
		
		// Add additional instances to distribution
		for(Instance i : neighbours){
			Instances b =  m_NNSearch.kNearestNeighbours(i, kcp*cls_num);
			for(Instance toadd : b){
				local.add(toadd);
			}
			local.add(i);
		}
		
		// Predict class
		NaiveBayes nb = new NaiveBayes();
		nb.buildClassifier(local);
		return nb.distributionForInstance(instance);
		
	}
	
	
	
}