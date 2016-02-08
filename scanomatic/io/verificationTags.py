def ctrlNum(s1, s2):
    """Methods returns a 1-3 digit control number.

    Algorithm was designed to notice misspellings in input strings.
    The algoritm should work well on any string pair, but was designed
    with for length strings containing only capital letters and numbers.
    If other characters are used, the return number size may become
    increased.

    :param s1: The first string-like ID-tag
    :param s2: The second string-like ID-tag
    :return: Verification integer in range 0-996
    """

    #First the two strings are concatenated
    s = s1 + s2

    #The max range for the return is set to 996 using the prime
    #997 in the modulus operation
    a = 997

    #Initiation of return value
    v = 1

    #For each character in s
    for i in range(len(s)):

        #If character position is an even position
        if i % 2 == 0:

            #Add the product of the character position and
            #the ordinal of the character to the return value
            v += i * ord(s[i])

        #If character position is an odd position
        else:

            #Multiply the return value with the ordinal of the character
            v *= ord(s[i])

        #Set the return value of the modulus of the return value and
        #the 'a' (range-max).
        v %= a

    return v
