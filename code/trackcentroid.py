# import the package compute distance between each pair of the two collections of inputs.
from scipy.spatial import distance as dist
# Return an instance of a dict subclass that has methods specialized for rearranging dictionary order.
from collections import OrderedDict
import numpy as np

class CentroidTracker():

    """ Initialize the maxdisappeared to some value to check after how many frames the object has disappeared."""
	

    def __init__(self, maxDisappeared=60):
        
        """
        Initialize the next unique object ID along with two ordered
        dictionaries used to keep track of mapping a given object
        ID to its centroid and number of consecutive frames it has been marked as "disappeared", respectively
        ----------
        Parameters
        ---------- 
        maxDisappeared: int
            store the number of maximum consecutive frames a given
            object is allowed to be marked as "disappeared" until we
            need to deregister the object from tracking
        Returns
        -------
            None
        """
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        """
        when registering an object we use the next available object
		ID to store the centroid 
        ----------
        Parameters
        ----------
        centroid: tuple (cX, cY)
            The centre of the bounding box is calculated using the starting and ending coordinates of the bounding box by taking the average.
        Returns
        -------
            None
        """
		
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        """
        to deregister an object ID we delete the object ID from both of our respective dictionaries        
        ----------
        Parameters
        ----------
        objectID: tuple (cX, cY)
            
        Returns
        -------
            None
        """

        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        """
        check to see if the list of input bounding box rectangles is empty        
        ----------
        Parameters
        ----------
        rects: list of numpy arrays
            contains the list of the coordinates of the bounding box obtained from object detector method.
        Returns
        -------
        objects: dictionary of object with key as objectID and element as coordinates of the bounding box
        """
        # Used to check the number of input bouding box rectangle is empty. It mean there is no object to track.
        if len(rects) == 0:
			# loop over any existing tracked objects and mark them as disappeared since object detector has detected zero rectangles.
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                # Check for every objecID in dissapeared dictionary to check if the disseparance value is greater
                # than the maximum threshold. That means after how many frames the object has disseapeared. If its true then 
                # degresiter that particular objectID since it is missing after checking maxdisseapred frames. 
				# (if we have reached a maximum number of consecutive frames where a given object has been marked as
				# missing, deregister it)
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

			# return the objects early as there are no centroids or tracking info to update
            return self.objects

        # initialize an array of input centroids for the detected bounding box in the current frame
    
        inputCentroids = np.zeros((len(rects), 2), dtype="int")

		# loop over the detected bounding box rectangles
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
			# use the bounding box coordinates to derive the centroid
            # Take the avergae to find the centroid
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

		# if we are currently not tracking any objects (It means objects is empty)take the input
		# centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])

		# otherwise, check how closely the detected object bounding box centroid is close to the 
        # existing obejct centroid by taking the euclidean distance. 
		# try to match the input centroids to existing object centroids
        else:
			# grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

			# compute the distance between each pair of existing object
			# centroids and input centroids, respectively. 
            # The distance is euclidean distance, found using distance in scipy.spatial method
			# The goal will be to match an input centroid to an existing
			# object centroid
            # It compute the euclidean distance between existing centroid and new detected centroid.
            # For ex. If there is two exisitng centroid so dimension is 2x2 [(x,y)*2](2 centroid) (number of row is the number of centroid detected)
            # And let's say 3 new object centroid has detected. So dimension is 3x2 (3 centroid)
            # Take the euclidean distance between the each existing centroid and new detected centroid so we get 2*3 = 6 euclidean distance. 
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

			# in order to perform this matching we must (1) find the
			# smallest value in each row and then (2) sort the row
			# indexes based on their minimum values so that the row
			# with the smallest value as at the *front* of the index
			# list
            # Below code will give us the row number that has minimum value in sorted fashion, 
            # means first find the small value in each row and now among those small values sort whihc row has the small value and return the row number.
            rows = D.min(axis=1).argsort()
            
            # In similar fashion we do for column.
            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list. Now we will get row and column number of D that has minimum value in sorted fashion
            cols = D.argmin(axis=1)[rows]
			# in order to determine if we need to update, register,
			# or deregister an object we need to keep track of which
			# of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()

			# loop over the combination of the (row, column) index
			# tuples
            for (row, col) in zip(rows, cols):
				# if we have already examined either the row or
				# column value before, ignore it
                if row in usedRows or col in usedCols:
                    continue

				# otherwise, grab the object ID for the current row,
				# set its new centroid, and reset the disappeared
				# counter
                # we update the object centroid
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

				# indicate that we have examined each of the row and
				# column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)

			# compute both the row and column index we have NOT yet
			# examined
            # we must determine which centroid indexes we haven’t examined yet and store them in two new convenient sets 
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

			# in the event that the number of object centroids is
			# equal or greater than the number of input centroids
			# we need to check and see if some of these objects have
			# potentially disappeared
            # final check handles any objects that have become lost or if they’ve potentially disappeared.
            if D.shape[0] >= D.shape[1]:
				# loop over the unused row indexes
                for row in unusedRows:
					# grab the object ID for the corresponding row
					# index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

					# check to see if the number of consecutive
					# frames the object has been marked "disappeared"
					# for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

			# otherwise, if the number of input centroids is greater
			# than the number of existing object centroids we need to
			# register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

		# return the set of trackable objects
        return self.objects
