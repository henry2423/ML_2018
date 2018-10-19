#include<iostream>
#include<stdio.h>
#include<vector>
#include<algorithm>
#include<queue>
#include<string.h>
#include<time.h>
#define MAX_TREE_NUM 15
#define MAX_TARGET_NUM 10
using namespace std;

/*-------------------------------------------------------------------------*
 *  The struct is for the single sample attributes   				       *
 *	feature : a vector to store features of single sample 			 	   *
 *  name : The target name of the feature vector                           *
 *  Take first line (5.1,3.5,1.4,0.2,Iris-setosa) of "iris.txt" as example:*
 *	feature:5.1,3.5,1.4,0.2 ; name: Iris-setosa                            *
 *-------------------------------------------------------------------------*/

struct SingleSample
{
	vector<double> feature;
	char name[20];
	
};
/*-------------------------------------------------------------------------*
 *  The struct is for the whole tree attributes   				           *
 *	SampleList : a vector to store number of SingleSample struct which is  *
 *				 used to build single tree	 	                           *
 *  RightList  : a vector to store number of SingleSample struct which is  *
 *				 derived by seperate the SampleList (use gini value)       *
 *  LeftList   : Same as RightList                                         *
 *	Gini       : Gini[0] for record the selected feature                   *
 *  depth      : Gini[1] for record threshod of the selected feature       *
 *  Left_purity: used to determine if the data is pure(1) or not(0)        *
 *  Right_purity: same as Left_purity                                      *
 *  Left, Right, Parent: point to the child / parent                       *
 *-------------------------------------------------------------------------*/
struct IrisTree{
	vector<struct SingleSample> SampleList;
	vector<struct SingleSample> RightList;
	vector<struct SingleSample> LeftList;
    double Gini[2];
    int depth;
	int Left_purity;
	int Right_purity;
    IrisTree *Left,*Right,*Parent;
};
/*-------------------------------------------------------------------------*
 *  Global variable which can be accessed everywhere				       *
 *  Following are important variable:                                      *
 *	result_list : a vector to store target name of sample (defined by user)*
 *  list1_target  :  to count the target amount for left list              *
 *  list2_target   : to count the target amount for Right list             *
 *-------------------------------------------------------------------------*/
int feature_index=0;
int feature_size_index=0;
IrisTree* root;
SingleSample* sample;

vector<char*> result_list;
int list1_target[MAX_TARGET_NUM];
int list2_target[MAX_TARGET_NUM];
/*used for user-define random_shuffle*/
int myrandom (int i) 
{ 
	return rand()%i;
}
/*used for sort, based on current feature index*/
bool cmp(SingleSample n1,SingleSample n2)
{ 
	return n1.feature[feature_size_index] < n2.feature[feature_size_index];
}
/*determine the if the sample list is pure or not */
int count_purity(vector<struct SingleSample> tmpList1, vector<struct SingleSample> tmpList2,int current_index)
{	
	if(current_index==0){ /*first access two list, need loop to get target number */
		for(int i=0; i<tmpList1.size();i++){
			for(int j=0 ; j<result_list.size();j++){		
				if(strcmp(tmpList1[i].name ,result_list[j]) ==0){ 
					list1_target[j]++;
					break;
				}	
			}
		}	
		for(int i=0;i<tmpList2.size();i++){
			for(int j=0 ; j<result_list.size();j++){
				if(strcmp(tmpList2[i].name ,result_list[j]) ==0){
					list2_target[j]++;
					break;
				}		
			}
		}
	}
	else{
		for(int j=0 ; j<result_list.size();j++){
			if(strcmp(tmpList1[current_index].name ,result_list[j]) ==0){
				list1_target[j]++;
				list2_target[j]--;
				break;
			}		
		}	
	}
	
	double num_list1 = 0.0 ; /*sum of the list_target*/
	double num_list2 = 0.0 ; /*                      */
	
	for(int i=0; i<result_list.size(); i++){
		num_list1+=list1_target[i];
		num_list2+=list2_target[i];
	}
	
	
	int purity_left = 0;
	int purity_right = 0;
	
	for(int i=0; i<result_list.size(); i++){
		if(num_list1 == list1_target[i]){
			purity_left=1;
			break;
		}
	} 
	for(int i=0; i<result_list.size(); i++){	
		if(num_list2 == list2_target[i]){
			purity_right = 1;
			break;
		}				
	}
	int purity =0; /*purity of the data set*/
	
	if(purity_left&&purity_right)
		purity=1;
	else if(purity_left)
		purity=2;	
	else if (purity_right)
		purity=3;
	else
		purity=0;
		
	purity_left=0;
	purity_right=0;
	
	return purity;
	
}
/*determin the gini of the data set*/
double count_Gini(int* list1_target, int* list2_target)
{	
	double num_list1 = 0.0 ;
	double num_list2 = 0.0 ;
	
	for(int i=0; i<result_list.size(); i++){
		num_list1+=list1_target[i];
		num_list2+=list2_target[i];
	}
	
	double total = num_list1 + num_list2;
	double gini_list1 = 1.0;
	double gini_list2 = 1.0;
	
	for(int i=0;i<result_list.size();i++){
		gini_list1 -= (list1_target[i]/num_list1)*(list1_target[i]/num_list1);
		gini_list2 -= (list2_target[i]/num_list2)*(list2_target[i]/num_list2);
	}
	
	double result = (num_list1/total)*gini_list1 + (num_list2/total)*gini_list2;
	
	return result;
}
/*classify the feature to correspond target determined by number of tree */
char* classify(vector<double> feature, IrisTree *node)
{		
	if(feature[int(node->Gini[0])] <= node->Gini[1]){
		if(node->Left ==NULL){
			int tmp_result[MAX_TARGET_NUM];
			for(int i=0;i<result_list.size();i++)
				tmp_result[i]=0;
				
			for(int i=0;i<node->LeftList.size();i++){
				for(int j=0;j<result_list.size();j++){
					if(strcmp(node->LeftList[i].name,result_list[j])==0){
						tmp_result[j]++;
						break;
					}
				}
			}
			for(int i=0;i<result_list.size();i++){
				/*classify it as the target which have the max number */
				if(tmp_result[i]== *(max_element(tmp_result,tmp_result+result_list.size()))){
					return result_list[i];
				}
			}			
		}		
		else
			return classify(feature, node->Left);
	}					
	else{
		if(node->Right == NULL){	
			int tmp_result[MAX_TARGET_NUM];
			for(int i=0;i<result_list.size();i++)
				tmp_result[i]=0;
				
			for(int i=0;i<node->LeftList.size();i++){
				for(int j=0;j<result_list.size();j++){
					if(strcmp(node->RightList[i].name,result_list[j])==0){
						tmp_result[j]++;
						break;
					}
				}
			}
			for(int i=0;i<result_list.size();i++){
				if(tmp_result[i]== *(max_element(tmp_result,tmp_result+result_list.size()))){
					return result_list[i];
				}
			}		
		}			
		else
			return classify(feature, node->Right);			
	}	
}
/*find the root of the tree*/
IrisTree *find_root(IrisTree *node)
{
	if(node->Parent==NULL)
		return node;
	
	find_root(node->Parent);
}
/*build the tree for the data set*/
void build_tree(IrisTree *node,int feature_size, int depth, int bagging_flag,int depth_limit_flag, vector<int> attr_chose)
{ 
	double min_gini = 1;
	double gini = 0;
	int purity = 0;
	int purity_record[2];
	int feature_record = 0;
	double threshold_record =0;
	vector<struct SingleSample> tmpList1_record;
	vector<struct SingleSample> tmpList2_record;
	cout<<"tree depth:"<<depth<<endl;
	for(int i=0;i<feature_size;i++){

		if(bagging_flag)
			feature_index= attr_chose[i];
		
		else
			feature_index = i;
		feature_size_index= i; /*for cmp func to know current feature index*/
		
		//sort the SampleList based on current feature 
		sort(node->SampleList.begin(),node->SampleList.end(),cmp); 
	
		for(int j=0;j<node->SampleList.size()-1;j++){
			if(node->SampleList[j].name == node->SampleList[j+1].name)
				continue; /* don't cout threshold unless the target is change*/
			double threshold = (node->SampleList[j].feature[i] + node->SampleList[j+1].feature[i])/2;
			vector<struct SingleSample> tmpList1;
			vector<struct SingleSample> tmpList2;
			tmpList1.assign(node->SampleList.begin(),node->SampleList.begin()+(j+1));
			tmpList2.assign(node->SampleList.begin()+(j+1),node->SampleList.end());
			
			purity = count_purity(tmpList1, tmpList2,j);
			gini = count_Gini(list1_target, list2_target);
			
			if(gini <= min_gini){ /*determine the best feature and threshold to apply*/
				min_gini = gini;
				feature_record = feature_index;
						
				tmpList1_record.assign(tmpList1.begin(),tmpList1.end());
				tmpList2_record.assign(tmpList2.begin(),tmpList2.end());
						
				threshold_record =threshold;
				purity_record[0] = (purity==1||purity==2)?1:0; /*left purity*/
				purity_record[1] = (purity==1||purity==3)?1:0; /*right purity*/
			}		
		}			
		for(int j=0 ; j<result_list.size();j++){ 
			/*need to be re-initializd since they will be re-count in next iteration*/
			list1_target[j]=0;
			list2_target[j]=0;
		}
	}
	
	node->Gini[0] = feature_record;
	node->Gini[1] = threshold_record;
	node->LeftList.assign(tmpList1_record.begin(),tmpList1_record.end());
	node->RightList.assign(tmpList2_record.begin(),tmpList2_record.end());
		
	if(purity_record[0]==0)
		node->Left_purity=0;
	else
		node->Left_purity=1;
	
	if(purity_record[1]==0)
		node->Right_purity=0;
	else
		node->Right_purity=1;
		
	if(depth_limit_flag==1&&depth==10) /*depth limit of the built tree*/
		return ;		
	else{ /*if the data is not pure, need to be separate again*/
		if(node->Left_purity == 0){
			root = new IrisTree;
			root->SampleList.assign(tmpList1_record.begin(),tmpList1_record.end());
			node->Left = root;
			root->Parent = node;
			root->Left =NULL;
			root->Right =NULL;
			root->depth = depth+1;

			build_tree(root, feature_size,root->depth,bagging_flag,depth_limit_flag,attr_chose);
		}			
		if(node->Right_purity==0){
			root = new IrisTree;
			root->SampleList.assign(tmpList2_record.begin(),tmpList2_record.end());
			node->Right = root;
			root->Parent = node;
			root->Left =NULL;
			root->Right =NULL;
			root->depth = depth+1;

			build_tree(root, feature_size, root->depth, bagging_flag,depth_limit_flag, attr_chose);
			
		}	
		return ;	
	}			
}

/*build the tree for the data set but use random-selected feature and threshold*/
void build_random_tree(IrisTree *node,int feature_size, int depth, int bagging_flag, int depth_limit_flag, vector<int> attr_chose)
{ 
	double gini = 0;
	int purity = 0;
	int purity_record[2];
	
	int rand_num;
	int rand_select_node;
	cout<<"tree depth:"<<depth<<endl;
	rand_num = rand()%feature_size; /*random select the feature*/
		if(bagging_flag)
			feature_index= attr_chose[rand_num];
		else
			feature_index = rand_num;
		feature_size_index= rand_num; /*for cmp func to know current feature index*/
		
		//sort the SampleList based on current feature 
		sort(node->SampleList.begin(),node->SampleList.end(),cmp); 
		rand_select_node = rand()%(node->SampleList.size()-1); /*random select the threshold*/
		
		double threshold = (node->SampleList[rand_select_node].feature[rand_num]+node->SampleList[rand_select_node+1].feature[rand_num])/2;
		
		vector<struct SingleSample> tmpList1;
		vector<struct SingleSample> tmpList2;
		tmpList1.assign(node->SampleList.begin(),node->SampleList.begin()+(rand_select_node+1));
		tmpList2.assign(node->SampleList.begin()+(rand_select_node+1),node->SampleList.end());
		
		purity = count_purity(tmpList1, tmpList2, 0);
		gini = count_Gini(list1_target, list2_target);
		
		purity_record[0] = (purity==1||purity==2)?1:0; /*left purity*/
		purity_record[1] = (purity==1||purity==3)?1:0; /*right purity*/
		for(int j=0 ; j<result_list.size();j++){ 
			/*need to be re-initializd since they will be re-count in next recursion*/
			list1_target[j]=0;
			list2_target[j]=0;
		}
				
	node->Gini[0] = feature_index;
	node->Gini[1] = threshold;
	node->LeftList.assign(tmpList1.begin(),tmpList1.end());
	node->RightList.assign(tmpList2.begin(),tmpList2.end());
		
	if(purity_record[0]==0)
		node->Left_purity=0;
	else
		node->Left_purity=1;
	
	if(purity_record[1]==0)
		node->Right_purity=0;
	else
		node->Right_purity=1;
		
	if(depth_limit_flag==1&&depth==10) /*depth limit of the built tree*/
		return ;		
	else{ /*if the data is not pure, need to be separate again*/
		if(node->Left_purity == 0){
			root = new IrisTree;
			root->SampleList.assign(tmpList1.begin(),tmpList1.end());
			node->Left = root;
			root->Parent = node;
			root->Left =NULL;
			root->Right =NULL;
			root->depth = depth+1;

			build_random_tree(root, feature_size,root->depth,bagging_flag,depth_limit_flag, attr_chose);
		}			
		if(node->Right_purity==0){
			root = new IrisTree;
			root->SampleList.assign(tmpList2.begin(),tmpList2.end());
			node->Right = root;
			root->Parent = node;
			root->Left =NULL;
			root->Right =NULL;
			root->depth = depth+1;

			build_random_tree(root, feature_size, root->depth, bagging_flag,depth_limit_flag, attr_chose);
			
		}	
		return ;	
	}			
}


int main(void)
{
	clock_t start_time,end_time; /*compute the execution time*/
	double cpu_time;
	
	FILE *file;
    char InFileName[30]={"iris.txt"}; /*user defined file name*/
	file=fopen(InFileName,"r");
    
    if(file==NULL)
        cout<<"Error Opening file"<<endl;  
		  
/*-------------------------------------------------------------------------*
 *  user-defined attribute                           				       *
 *	tree_num : the number of tree in the forest      	                   *
 *  attr_bagging_flag : attribute bagging or not                           *
 *  total_feature_num : total feature number of a data                     *
 *  Take "iris.txt" as example- total_feature_num is 4                     *                           
 *-------------------------------------------------------------------------*/
	int tree_num=0;
	int attr_bagging_flag=0;
	int total_feature_num=0;
	int extremely_random_flag=0;
	int depth_limit_flag=0;
	
/*-------------------------------------------------------------------------*
 *	list : all the sample      	                                           *
 *  train : retrieve in list (75%)                                         *
 *  validate : retrieve in list (25%)                                      *
 *  attr_bagging_train:  retrieve in list with atribute bagging            *
 *  extremely_random_flag:determine if build extremely random forest or not*
 *  depth_limit_flag: do the depth limit or not                            *
 *-------------------------------------------------------------------------*/	
	IrisTree *Root[MAX_TREE_NUM] ;
    struct SingleSample sample;
    vector<struct SingleSample> list;
    vector<struct SingleSample> train;
    vector<struct SingleSample> validate;
    vector<struct SingleSample> attr_bagging_train; 
    
    /*user dedined attribute*/
    if(strcmp("iris.txt",InFileName)==0)
    {
    	result_list.push_back("Iris-setosa");
    	result_list.push_back("Iris-versicolor");
    	result_list.push_back("Iris-virginica");
    	
    	total_feature_num = 4;
    	tree_num = 20;
    	attr_bagging_flag=0;
    	extremely_random_flag = 0;
    	depth_limit_flag = 0;
	}
	else if(strcmp("cross200.txt",InFileName)==0)
	{
		result_list.push_back("1");
		result_list.push_back("2");	
		
		total_feature_num=2;
		tree_num = 20;
		attr_bagging_flag=1;
		extremely_random_flag = 0;
		depth_limit_flag =1;
	}
	else if(strcmp("optical-digits.txt",InFileName)==0)
	{
		result_list.push_back("1");
		result_list.push_back("2");	
		result_list.push_back("3");
		result_list.push_back("4");	
		result_list.push_back("5");
		result_list.push_back("6");	
		result_list.push_back("7");
		result_list.push_back("8");	
		result_list.push_back("9");
		result_list.push_back("0");	
			
		total_feature_num=64;
		tree_num = 20;
		attr_bagging_flag=1;
		extremely_random_flag = 0;
		depth_limit_flag =1;
	}

    while (!feof(file)){
    	start_time = clock();
	   	for(int i=0;i<total_feature_num+1;i++){
	   		if(i==total_feature_num){
	   			char name[2];
				fscanf(file,"%s\n",&sample.name);
			}
	   		else{	
	   			double attr;
	   			fscanf( file,"%lf,", &attr);
	   			sample.feature.push_back(attr);
			}
		}	        
      list.push_back(sample);
      sample.feature.clear();
    }
	srand(time(0));
    random_shuffle(list.begin(),list.end(),myrandom);/*random the lsit*/
  	
    validate.assign(list.begin(),list.begin()+list.size()*0.25); /*distribut the list*/
    train.assign(list.begin()+list.size()*0.25,list.end());
    
	for(int i=0 ; i<result_list.size();i++){
		list1_target[i]=0;
		list2_target[i]=0;
	}

	vector<int>	attr_chose; /*for attribute bagging*/
	for(int i=0;i<total_feature_num;i++)
		attr_chose.push_back(i);
	
    cout<<"Start to build forest..."<<endl;
	for(int i=0;i<tree_num;i++){
		if(attr_bagging_flag==1){
			int bag_num = int(sqrt(total_feature_num));/*bagging size*/
			/*random the attr_chose vector to do the attribute bagging at each iteration*/
			random_shuffle(attr_chose.begin(), attr_chose.end()); 
			
			attr_bagging_train.clear();
			for(int j=0;j<train.size();j++){
				struct SingleSample attr_sample;
				
				for(int k=0;k<bag_num;k++)
					attr_sample.feature.push_back(train[j].feature[attr_chose[k]]);
				
				strcpy(attr_sample.name,train[j].name); 	
				attr_bagging_train.push_back(attr_sample);
			}
			
			Root[i] = new IrisTree;
			Root[i]->SampleList.assign(attr_bagging_train.begin(),attr_bagging_train.begin()+attr_bagging_train.size()*0.4);
    		Root[i] ->Left = NULL;
    		Root[i] ->Right = NULL;
    		Root[i] ->Parent = NULL;
    		Root[i] ->depth = 1;
    		
    		if(extremely_random_flag==1)
    			build_random_tree(Root[i], bag_num, Root[i] ->depth, attr_bagging_flag, depth_limit_flag,attr_chose);
    		else
    			build_tree(Root[i], bag_num,Root[i] ->depth, attr_bagging_flag, depth_limit_flag, attr_chose);
    		Root[i] = find_root(Root[i]);
    		/*after each iteration, need to re-random the train list to do the tree bagging*/
			random_shuffle(train.begin(),train.end(),myrandom);			
		}
		else{
			Root[i] = new IrisTree;
			Root[i]->SampleList.assign(train.begin(),train.begin()+train.size()*0.4);
    		Root[i] ->Left =NULL;
    		Root[i] ->Right = NULL;
    		Root[i] ->Parent = NULL;
    		Root[i] ->depth = 1;
    		if(extremely_random_flag==1)
    			build_random_tree(Root[i], total_feature_num, Root[i] ->depth, attr_bagging_flag, depth_limit_flag ,attr_chose);
    		else 
    			build_tree(Root[i], total_feature_num, Root[i] ->depth, attr_bagging_flag, depth_limit_flag, attr_chose);
    		Root[i] = find_root(Root[i]);

    		random_shuffle(train.begin(),train.end(),myrandom);
		}
	cout<<endl;   	
	}
	cout<<"Build forest finish!"<<endl;
    
    double accuracy=0;
    int result[MAX_TARGET_NUM];
    
    for(int i=0;i<validate.size();i++){
    	for(int j=0;j<result_list.size();j++)
    		result[j]=0;
		/*classify the validation data*/
		for(int j=0 ; j<tree_num;j++){
			for(int k=0;k<result_list.size();k++){ 
			    if(strcmp(classify(validate[i].feature, Root[j]) ,result_list[k]) == 0){
						result[k]++;
						break;
				}			
			} 	
		}
		/*compute the accuray*/
		for(int j=0;j<result_list.size();j++){
			if(result[j]== *(max_element(result,result+result_list.size()))){
				
				if(strcmp(result_list[j], validate[i].name)==0){
					accuracy++;
					break;
				}
					
			}
		}		
	}
	end_time = clock();
	cpu_time = (double)(((double) (end_time - start_time)) / CLOCKS_PER_SEC);
				cout<<"Execution time:  "<<cpu_time<<" sec"<<endl<<endl;
	cout<<"Validate accuracy:"<<accuracy/validate.size()<<endl;
    
    fclose(file);
        
	return 0;
	
}
