import pandas as pd

inputTrain = '../RawData/train.csv'
inputTest = '../RawData/test.csv'
outputTrain = '../ProcessedData/train.csv'
outputTest = '../ProcessedData/test.csv'

train = pd.read_csv(inputTrain, header=0)
test = pd.read_csv(inputTest, header=0)

avgAgeMaster = train.loc[train['Name'].str.contains(', Master. '), 'Age'].mean()
avgAgeMr = train.loc[train['Name'].str.contains(', Mr. '), 'Age'].mean()
avgAgeMrs = train.loc[train['Name'].str.contains(', Mrs. '), 'Age'].mean()
avgAgeMiss = train.loc[train['Name'].str.contains(', Miss. '), 'Age'].mean()
meanFare = train['Fare'].mean()


def processData(frame, outputURL):
    frame['ageUnknown'] = frame['Age'].isnull()
    frame['cabinUnknown'] = frame['Cabin'].isnull()
    
    frame['isMaster'] = frame['Name'].str.contains(', Master. ')
    frame.loc[(frame['isMaster'] == True) & (frame['Age'].isnull()), 'Age'] = avgAgeMaster
    
    frame['isMr'] = frame['Name'].str.contains(', Mr. ')
    frame.loc[(frame['isMr'] == True) & (frame['Age'].isnull()), 'Age'] = avgAgeMr
    frame.loc[(frame['Name'].str.contains(', Dr. ')) & (frame['Sex'] == 'male'), 'Age'] = avgAgeMr
    
    frame['isMrs'] = (frame['Name'].str.contains(', Mrs. ') | (frame['Name'].str.contains(', Dr. ')) & (frame['Sex'] == 'female'))
    frame.loc[((frame['isMrs'] == True) | (frame['Name'].str.contains(', Ms. '))) & (frame['Age'].isnull()), 'Age'] = avgAgeMrs
    #frame.loc[(frame['Name'].str.contains(', Dr. ')) & (frame['Sex'] == 'female'), 'Age'] = avgAgeMrs
    
    frame['isMiss'] = frame['Name'].str.contains(', Miss.')
    frame.loc[(frame['isMiss'] == True) & (frame['Age'].isnull()), 'Age'] = avgAgeMiss
    
    frame['hasNickname'] = frame['Name'].str.contains('\"')
    
    frame['hasTitle'] = ((frame['Name'].str.contains(', Dr. ') & (frame['Name'].str.contains(', Ms. ') == False)) | ((frame['isMr'] == False) & (frame['isMrs'] == False) & (frame['isMaster'] == False) & (frame['isMiss'] == False)))

    frame['A_Deck'] = False
    frame['B_Deck'] = False
    frame['C_Deck'] = False
    frame['D_Deck'] = False
    frame['E_Deck'] = False
    frame['F_Deck'] = False
    frame['G_Deck'] = False

    frame.loc[frame['cabinUnknown'] == False, 'A_Deck'] = frame['Cabin'].str.contains('A')
    frame.loc[frame['cabinUnknown'] == False, 'B_Deck'] = frame['Cabin'].str.contains('B')
    frame.loc[frame['cabinUnknown'] == False, 'C_Deck'] = frame['Cabin'].str.contains('C')
    frame.loc[frame['cabinUnknown'] == False, 'D_Deck'] = frame['Cabin'].str.contains('D')
    frame.loc[frame['cabinUnknown'] == False, 'E_Deck'] = frame['Cabin'].str.contains('E')
    frame.loc[frame['cabinUnknown'] == False, 'F_Deck'] = frame['Cabin'].str.contains('F')
    frame.loc[frame['cabinUnknown'] == False, 'G_Deck'] = frame['Cabin'].str.contains('G')
    
    frame = pd.get_dummies(frame, columns=['Embarked'])
    frame['isMale'] = (frame['Sex'] == 'male')
    frame['isChild'] = frame['Age'] <= 12
    frame['isAdult'] = frame['Age'] >= 18
    frame['upperClassChild'] = (frame['isChild'] == True) & (frame['Pclass'] < 3)
    
    del frame['Sex']
    
    frame.loc[frame['Fare'].isnull(), 'Fare'] = float(meanFare)
    
    del frame['Name']
    del frame['Cabin']
    del frame['Ticket']
    del frame['isMaster']
    del frame['isMr']
    del frame['isMiss']
    del frame['isMrs']
    
    # remove cabin text up to and including first space
    '''ticketColumn = frame['Ticket']
    for i in range(frame.shape[0]):
        if " " in ticketColumn[i]:
            spaceIndex = ticketColumn[i].index(" ")
            if " " in ticketColumn[i][spaceIndex+1:]:
                spaceIndex2 = ticketColumn[i][spaceIndex+1:].index(" ")
                print(spaceIndex2)
                ticketColumn.iloc[i] = float(ticketColumn.iloc[i][spaceIndex+spaceIndex2+1:])
            else:
                ticketColumn.iloc[i] = float(ticketColumn.iloc[i][spaceIndex+1:])
    
    frame['Ticket'] = ticketColumn'''
            
    frame.to_csv(outputURL, index=False)
    print("Done!")
    print(frame.info())

processData(train, outputTrain)
processData(test, outputTest)