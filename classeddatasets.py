"""
Created Tuesday 11/17/2020 08:30 AM PST

Authors:
  Alexander Guyer

Description:
  Provides a class used to filter a dataset by labels. Useful for constructing
  class splits.
"""

from torch.utils.data import Dataset, Subset
from typing import Iterable, Dict

"""
Description:
   Class used to filter a dataset by labels.
   
Attributes:
  dataset: The underlying filtered dataset.
  labels: The original labels whose data points are retained post-filter
          (whitelisting).
  label_mapping: Dictionary mapping the original (unfiltered) labels to the new
                 filtered labels. This is necessary for maintaining label
		 consistency across multiple ClassedDatasets.
"""
class ClassedDataset(object):
	"""
	Description:
	  Constructor

	Parameters:
	  dataset: Dataset to filter.
	  labels: Whitelisting labels whose data points are retained
	          post-filter.
	  label_mapping: Optional label mapping from original labels to the new
	                 filtered labels.
	"""
	def __init__(self, dataset: Dataset, labels: Iterable[int],
	             label_mapping: Dict[int, int] = None):
		assert (labels is not None and len(labels) > 0), \
		        'labels must not be NoneType or empty'
		
		# If not provided, create a mapping from original label range
		# to new label range:
		
		if label_mapping is None:
			label_mapping = {}
			new_label = 0
			for label in labels:
				label_mapping[label] = new_label
				new_label += 1
		
		self.label_mapping = label_mapping

		indices = []
		for idx, (_, label) in enumerate(dataset):
			if label in labels:
				indices.append(idx)
		
		self.dataset = Subset(dataset, indices)
	
	"""
	Description:
	  Returns the number of elements in the filtered dataset
	"""
	def __len__(self):
		return len(self.dataset)
	
	"""
	Description:
	  Returns the element in the filtered dataset at the provided index
	
	Parameters:
	  index: Index of element to retrieve
	"""
	def __getitem__(self, index: int):
		data, label = self.dataset.__getitem__(index)
		label = self.label_mapping[label]
		return data, label
