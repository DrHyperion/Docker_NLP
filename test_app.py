import unittest
import os
import requests

class FlaskTests(unittest.TestCase):

	def setUp(self):
		os.environ['NO_PROXY'] = '0.0.0.0'
		self.user = {
			'message': "hi"
		}
		pass
		
	def tearDown(self):
		pass
	
	
	def test_a_connection(self):
		responce = requests.get('http://localhost:4000')
		self.assertEqual(responce.status_code, 200)
		
	def test_b_prediction_positive(self):
		
		params = [
			{'message': "hi"},
			{'message': "love you"},
			{'message': "the sun"},
			{'message': "good play"},
			{'message': "nice home"}
			
			]
		for i in params:
			responce = requests.post('http://localhost:4000/predict', data=i)
			print(i)
			self.assertIn("Positive",str(responce.content))
	
	def test_b_prediction_negative(self):
		
		params = [
			{'message': "cry"},
			{'message': "hate this"},
			{'message': "bad habits"},
			{'message': "abandon"},
			{'message': "war"}
			
			]
		for i in params:
			responce = requests.post('http://localhost:4000/predict', data=i)
			print(i)
			self.assertIn("Negative",str(responce.content))
	
				
		
if __name__ == '__main__':
	unittest.main()		




