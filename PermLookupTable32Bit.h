#pragma once
#include <cstdint>

class PermLookupTable32Bit {
public:
    // shamelessly copied from https://github.com/damageboy/VxSort/blob/master/VxSort/BytePermutationTables.cs 
    // and modified to match my implementation
    static constexpr uint32_t permTable[256][8] = {
        {0, 1, 2, 3, 4, 5, 6, 7}, // 0b00000000 (0)|Left-PC: 8
        {1, 2, 3, 4, 5, 6, 7, 0}, // 0b00000001 (1)|Left-PC: 7    
        {0, 2, 3, 4, 5, 6, 7, 1}, // 0b00000010 (2)|Left-PC: 7    
        {2, 3, 4, 5, 6, 7, 0, 1}, // 0b00000011 (3)|Left-PC: 6
        {0, 1, 3, 4, 5, 6, 7, 2}, // 0b00000100 (4)|Left-PC: 7
        {1, 3, 4, 5, 6, 7, 0, 2}, // 0b00000101 (5)|Left-PC: 6
        {0, 3, 4, 5, 6, 7, 1, 2}, // 0b00000110 (6)|Left-PC: 6
        {3, 4, 5, 6, 7, 0, 1, 2}, // 0b00000111 (7)|Left-PC: 5
        {0, 1, 2, 4, 5, 6, 7, 3}, // 0b00001000 (8)|Left-PC: 7
        {1, 2, 4, 5, 6, 7, 0, 3}, // 0b00001001 (9)|Left-PC: 6
        {0, 2, 4, 5, 6, 7, 1, 3}, // 0b00001010 (10)|Left-PC: 6
        {2, 4, 5, 6, 7, 0, 1, 3}, // 0b00001011 (11)|Left-PC: 5
        {0, 1, 4, 5, 6, 7, 2, 3}, // 0b00001100 (12)|Left-PC: 6
        {1, 4, 5, 6, 7, 0, 2, 3}, // 0b00001101 (13)|Left-PC: 5
        {0, 4, 5, 6, 7, 1, 2, 3}, // 0b00001110 (14)|Left-PC: 5
        {4, 5, 6, 7, 0, 1, 2, 3}, // 0b00001111 (15)|Left-PC: 4
        {0, 1, 2, 3, 5, 6, 7, 4}, // 0b00010000 (16)|Left-PC: 7
        {1, 2, 3, 5, 6, 7, 0, 4}, // 0b00010001 (17)|Left-PC: 6
        {0, 2, 3, 5, 6, 7, 1, 4}, // 0b00010010 (18)|Left-PC: 6
        {2, 3, 5, 6, 7, 0, 1, 4}, // 0b00010011 (19)|Left-PC: 5
        {0, 1, 3, 5, 6, 7, 2, 4}, // 0b00010100 (20)|Left-PC: 6
        {1, 3, 5, 6, 7, 0, 2, 4}, // 0b00010101 (21)|Left-PC: 5
        {0, 3, 5, 6, 7, 1, 2, 4}, // 0b00010110 (22)|Left-PC: 5
        {3, 5, 6, 7, 0, 1, 2, 4}, // 0b00010111 (23)|Left-PC: 4
        {0, 1, 2, 5, 6, 7, 3, 4}, // 0b00011000 (24)|Left-PC: 6
        {1, 2, 5, 6, 7, 0, 3, 4}, // 0b00011001 (25)|Left-PC: 5
        {0, 2, 5, 6, 7, 1, 3, 4}, // 0b00011010 (26)|Left-PC: 5
        {2, 5, 6, 7, 0, 1, 3, 4}, // 0b00011011 (27)|Left-PC: 4
        {0, 1, 5, 6, 7, 2, 3, 4}, // 0b00011100 (28)|Left-PC: 5
        {1, 5, 6, 7, 0, 2, 3, 4}, // 0b00011101 (29)|Left-PC: 4
        {0, 5, 6, 7, 1, 2, 3, 4}, // 0b00011110 (30)|Left-PC: 4
        {5, 6, 7, 0, 1, 2, 3, 4}, // 0b00011111 (31)|Left-PC: 3
        {0, 1, 2, 3, 4, 6, 7, 5}, // 0b00100000 (32)|Left-PC: 7
        {1, 2, 3, 4, 6, 7, 0, 5}, // 0b00100001 (33)|Left-PC: 6
        {0, 2, 3, 4, 6, 7, 1, 5}, // 0b00100010 (34)|Left-PC: 6
        {2, 3, 4, 6, 7, 0, 1, 5}, // 0b00100011 (35)|Left-PC: 5
        {0, 1, 3, 4, 6, 7, 2, 5}, // 0b00100100 (36)|Left-PC: 6
        {1, 3, 4, 6, 7, 0, 2, 5}, // 0b00100101 (37)|Left-PC: 5
        {0, 3, 4, 6, 7, 1, 2, 5}, // 0b00100110 (38)|Left-PC: 5
        {3, 4, 6, 7, 0, 1, 2, 5}, // 0b00100111 (39)|Left-PC: 4
        {0, 1, 2, 4, 6, 7, 3, 5}, // 0b00101000 (40)|Left-PC: 6
        {1, 2, 4, 6, 7, 0, 3, 5}, // 0b00101001 (41)|Left-PC: 5
        {0, 2, 4, 6, 7, 1, 3, 5}, // 0b00101010 (42)|Left-PC: 5
        {2, 4, 6, 7, 0, 1, 3, 5}, // 0b00101011 (43)|Left-PC: 4
        {0, 1, 4, 6, 7, 2, 3, 5}, // 0b00101100 (44)|Left-PC: 5
        {1, 4, 6, 7, 0, 2, 3, 5}, // 0b00101101 (45)|Left-PC: 4
        {0, 4, 6, 7, 1, 2, 3, 5}, // 0b00101110 (46)|Left-PC: 4
        {4, 6, 7, 0, 1, 2, 3, 5}, // 0b00101111 (47)|Left-PC: 3
        {0, 1, 2, 3, 6, 7, 4, 5}, // 0b00110000 (48)|Left-PC: 6
        {1, 2, 3, 6, 7, 0, 4, 5}, // 0b00110001 (49)|Left-PC: 5
        {0, 2, 3, 6, 7, 1, 4, 5}, // 0b00110010 (50)|Left-PC: 5
        {2, 3, 6, 7, 0, 1, 4, 5}, // 0b00110011 (51)|Left-PC: 4
        {0, 1, 3, 6, 7, 2, 4, 5}, // 0b00110100 (52)|Left-PC: 5
        {1, 3, 6, 7, 0, 2, 4, 5}, // 0b00110101 (53)|Left-PC: 4
        {0, 3, 6, 7, 1, 2, 4, 5}, // 0b00110110 (54)|Left-PC: 4
        {3, 6, 7, 0, 1, 2, 4, 5}, // 0b00110111 (55)|Left-PC: 3
        {0, 1, 2, 6, 7, 3, 4, 5}, // 0b00111000 (56)|Left-PC: 5
        {1, 2, 6, 7, 0, 3, 4, 5}, // 0b00111001 (57)|Left-PC: 4
        {0, 2, 6, 7, 1, 3, 4, 5}, // 0b00111010 (58)|Left-PC: 4
        {2, 6, 7, 0, 1, 3, 4, 5}, // 0b00111011 (59)|Left-PC: 3
        {0, 1, 6, 7, 2, 3, 4, 5}, // 0b00111100 (60)|Left-PC: 4
        {1, 6, 7, 0, 2, 3, 4, 5}, // 0b00111101 (61)|Left-PC: 3
        {0, 6, 7, 1, 2, 3, 4, 5}, // 0b00111110 (62)|Left-PC: 3
        {6, 7, 0, 1, 2, 3, 4, 5}, // 0b00111111 (63)|Left-PC: 2
        {0, 1, 2, 3, 4, 5, 7, 6}, // 0b01000000 (64)|Left-PC: 7
        {1, 2, 3, 4, 5, 7, 0, 6}, // 0b01000001 (65)|Left-PC: 6
        {0, 2, 3, 4, 5, 7, 1, 6}, // 0b01000010 (66)|Left-PC: 6
        {2, 3, 4, 5, 7, 0, 1, 6}, // 0b01000011 (67)|Left-PC: 5
        {0, 1, 3, 4, 5, 7, 2, 6}, // 0b01000100 (68)|Left-PC: 6
        {1, 3, 4, 5, 7, 0, 2, 6}, // 0b01000101 (69)|Left-PC: 5
        {0, 3, 4, 5, 7, 1, 2, 6}, // 0b01000110 (70)|Left-PC: 5
        {3, 4, 5, 7, 0, 1, 2, 6}, // 0b01000111 (71)|Left-PC: 4
        {0, 1, 2, 4, 5, 7, 3, 6}, // 0b01001000 (72)|Left-PC: 6
        {1, 2, 4, 5, 7, 0, 3, 6}, // 0b01001001 (73)|Left-PC: 5
        {0, 2, 4, 5, 7, 1, 3, 6}, // 0b01001010 (74)|Left-PC: 5
        {2, 4, 5, 7, 0, 1, 3, 6}, // 0b01001011 (75)|Left-PC: 4
        {0, 1, 4, 5, 7, 2, 3, 6}, // 0b01001100 (76)|Left-PC: 5
        {1, 4, 5, 7, 0, 2, 3, 6}, // 0b01001101 (77)|Left-PC: 4
        {0, 4, 5, 7, 1, 2, 3, 6}, // 0b01001110 (78)|Left-PC: 4
        {4, 5, 7, 0, 1, 2, 3, 6}, // 0b01001111 (79)|Left-PC: 3
        {0, 1, 2, 3, 5, 7, 4, 6}, // 0b01010000 (80)|Left-PC: 6
        {1, 2, 3, 5, 7, 0, 4, 6}, // 0b01010001 (81)|Left-PC: 5
        {0, 2, 3, 5, 7, 1, 4, 6}, // 0b01010010 (82)|Left-PC: 5
        {2, 3, 5, 7, 0, 1, 4, 6}, // 0b01010011 (83)|Left-PC: 4
        {0, 1, 3, 5, 7, 2, 4, 6}, // 0b01010100 (84)|Left-PC: 5
        {1, 3, 5, 7, 0, 2, 4, 6}, // 0b01010101 (85)|Left-PC: 4
        {0, 3, 5, 7, 1, 2, 4, 6}, // 0b01010110 (86)|Left-PC: 4
        {3, 5, 7, 0, 1, 2, 4, 6}, // 0b01010111 (87)|Left-PC: 3
        {0, 1, 2, 5, 7, 3, 4, 6}, // 0b01011000 (88)|Left-PC: 5
        {1, 2, 5, 7, 0, 3, 4, 6}, // 0b01011001 (89)|Left-PC: 4
        {0, 2, 5, 7, 1, 3, 4, 6}, // 0b01011010 (90)|Left-PC: 4
        {2, 5, 7, 0, 1, 3, 4, 6}, // 0b01011011 (91)|Left-PC: 3
        {0, 1, 5, 7, 2, 3, 4, 6}, // 0b01011100 (92)|Left-PC: 4
        {1, 5, 7, 0, 2, 3, 4, 6}, // 0b01011101 (93)|Left-PC: 3
        {0, 5, 7, 1, 2, 3, 4, 6}, // 0b01011110 (94)|Left-PC: 3
        {5, 7, 0, 1, 2, 3, 4, 6}, // 0b01011111 (95)|Left-PC: 2
        {0, 1, 2, 3, 4, 7, 5, 6}, // 0b01100000 (96)|Left-PC: 6
        {1, 2, 3, 4, 7, 0, 5, 6}, // 0b01100001 (97)|Left-PC: 5
        {0, 2, 3, 4, 7, 1, 5, 6}, // 0b01100010 (98)|Left-PC: 5
        {2, 3, 4, 7, 0, 1, 5, 6}, // 0b01100011 (99)|Left-PC: 4
        {0, 1, 3, 4, 7, 2, 5, 6}, // 0b01100100 (100)|Left-PC: 5
        {1, 3, 4, 7, 0, 2, 5, 6}, // 0b01100101 (101)|Left-PC: 4
        {0, 3, 4, 7, 1, 2, 5, 6}, // 0b01100110 (102)|Left-PC: 4
        {3, 4, 7, 0, 1, 2, 5, 6}, // 0b01100111 (103)|Left-PC: 3
        {0, 1, 2, 4, 7, 3, 5, 6}, // 0b01101000 (104)|Left-PC: 5
        {1, 2, 4, 7, 0, 3, 5, 6}, // 0b01101001 (105)|Left-PC: 4
        {0, 2, 4, 7, 1, 3, 5, 6}, // 0b01101010 (106)|Left-PC: 4
        {2, 4, 7, 0, 1, 3, 5, 6}, // 0b01101011 (107)|Left-PC: 3
        {0, 1, 4, 7, 2, 3, 5, 6}, // 0b01101100 (108)|Left-PC: 4
        {1, 4, 7, 0, 2, 3, 5, 6}, // 0b01101101 (109)|Left-PC: 3
        {0, 4, 7, 1, 2, 3, 5, 6}, // 0b01101110 (110)|Left-PC: 3
        {4, 7, 0, 1, 2, 3, 5, 6}, // 0b01101111 (111)|Left-PC: 2
        {0, 1, 2, 3, 7, 4, 5, 6}, // 0b01110000 (112)|Left-PC: 5
        {1, 2, 3, 7, 0, 4, 5, 6}, // 0b01110001 (113)|Left-PC: 4
        {0, 2, 3, 7, 1, 4, 5, 6}, // 0b01110010 (114)|Left-PC: 4
        {2, 3, 7, 0, 1, 4, 5, 6}, // 0b01110011 (115)|Left-PC: 3
        {0, 1, 3, 7, 2, 4, 5, 6}, // 0b01110100 (116)|Left-PC: 4
        {1, 3, 7, 0, 2, 4, 5, 6}, // 0b01110101 (117)|Left-PC: 3
        {0, 3, 7, 1, 2, 4, 5, 6}, // 0b01110110 (118)|Left-PC: 3
        {3, 7, 0, 1, 2, 4, 5, 6}, // 0b01110111 (119)|Left-PC: 2
        {0, 1, 2, 7, 3, 4, 5, 6}, // 0b01111000 (120)|Left-PC: 4
        {1, 2, 7, 0, 3, 4, 5, 6}, // 0b01111001 (121)|Left-PC: 3
        {0, 2, 7, 1, 3, 4, 5, 6}, // 0b01111010 (122)|Left-PC: 3
        {2, 7, 0, 1, 3, 4, 5, 6}, // 0b01111011 (123)|Left-PC: 2
        {0, 1, 7, 2, 3, 4, 5, 6}, // 0b01111100 (124)|Left-PC: 3
        {1, 7, 0, 2, 3, 4, 5, 6}, // 0b01111101 (125)|Left-PC: 2
        {0, 7, 1, 2, 3, 4, 5, 6}, // 0b01111110 (126)|Left-PC: 2
        {7, 0, 1, 2, 3, 4, 5, 6}, // 0b01111111 (127)|Left-PC: 1
        {0, 1, 2, 3, 4, 5, 6, 7}, // 0b10000000 (128)|Left-PC: 7
        {1, 2, 3, 4, 5, 6, 0, 7}, // 0b10000001 (129)|Left-PC: 6
        {0, 2, 3, 4, 5, 6, 1, 7}, // 0b10000010 (130)|Left-PC: 6
        {2, 3, 4, 5, 6, 0, 1, 7}, // 0b10000011 (131)|Left-PC: 5
        {0, 1, 3, 4, 5, 6, 2, 7}, // 0b10000100 (132)|Left-PC: 6
        {1, 3, 4, 5, 6, 0, 2, 7}, // 0b10000101 (133)|Left-PC: 5
        {0, 3, 4, 5, 6, 1, 2, 7}, // 0b10000110 (134)|Left-PC: 5
        {3, 4, 5, 6, 0, 1, 2, 7}, // 0b10000111 (135)|Left-PC: 4
        {0, 1, 2, 4, 5, 6, 3, 7}, // 0b10001000 (136)|Left-PC: 6
        {1, 2, 4, 5, 6, 0, 3, 7}, // 0b10001001 (137)|Left-PC: 5
        {0, 2, 4, 5, 6, 1, 3, 7}, // 0b10001010 (138)|Left-PC: 5
        {2, 4, 5, 6, 0, 1, 3, 7}, // 0b10001011 (139)|Left-PC: 4
        {0, 1, 4, 5, 6, 2, 3, 7}, // 0b10001100 (140)|Left-PC: 5
        {1, 4, 5, 6, 0, 2, 3, 7}, // 0b10001101 (141)|Left-PC: 4
        {0, 4, 5, 6, 1, 2, 3, 7}, // 0b10001110 (142)|Left-PC: 4
        {4, 5, 6, 0, 1, 2, 3, 7}, // 0b10001111 (143)|Left-PC: 3
        {0, 1, 2, 3, 5, 6, 4, 7}, // 0b10010000 (144)|Left-PC: 6
        {1, 2, 3, 5, 6, 0, 4, 7}, // 0b10010001 (145)|Left-PC: 5
        {0, 2, 3, 5, 6, 1, 4, 7}, // 0b10010010 (146)|Left-PC: 5
        {2, 3, 5, 6, 0, 1, 4, 7}, // 0b10010011 (147)|Left-PC: 4
        {0, 1, 3, 5, 6, 2, 4, 7}, // 0b10010100 (148)|Left-PC: 5
        {1, 3, 5, 6, 0, 2, 4, 7}, // 0b10010101 (149)|Left-PC: 4
        {0, 3, 5, 6, 1, 2, 4, 7}, // 0b10010110 (150)|Left-PC: 4
        {3, 5, 6, 0, 1, 2, 4, 7}, // 0b10010111 (151)|Left-PC: 3
        {0, 1, 2, 5, 6, 3, 4, 7}, // 0b10011000 (152)|Left-PC: 5
        {1, 2, 5, 6, 0, 3, 4, 7}, // 0b10011001 (153)|Left-PC: 4
        {0, 2, 5, 6, 1, 3, 4, 7}, // 0b10011010 (154)|Left-PC: 4
        {2, 5, 6, 0, 1, 3, 4, 7}, // 0b10011011 (155)|Left-PC: 3
        {0, 1, 5, 6, 2, 3, 4, 7}, // 0b10011100 (156)|Left-PC: 4
        {1, 5, 6, 0, 2, 3, 4, 7}, // 0b10011101 (157)|Left-PC: 3
        {0, 5, 6, 1, 2, 3, 4, 7}, // 0b10011110 (158)|Left-PC: 3
        {5, 6, 0, 1, 2, 3, 4, 7}, // 0b10011111 (159)|Left-PC: 2
        {0, 1, 2, 3, 4, 6, 5, 7}, // 0b10100000 (160)|Left-PC: 6
        {1, 2, 3, 4, 6, 0, 5, 7}, // 0b10100001 (161)|Left-PC: 5
        {0, 2, 3, 4, 6, 1, 5, 7}, // 0b10100010 (162)|Left-PC: 5
        {2, 3, 4, 6, 0, 1, 5, 7}, // 0b10100011 (163)|Left-PC: 4
        {0, 1, 3, 4, 6, 2, 5, 7}, // 0b10100100 (164)|Left-PC: 5
        {1, 3, 4, 6, 0, 2, 5, 7}, // 0b10100101 (165)|Left-PC: 4
        {0, 3, 4, 6, 1, 2, 5, 7}, // 0b10100110 (166)|Left-PC: 4
        {3, 4, 6, 0, 1, 2, 5, 7}, // 0b10100111 (167)|Left-PC: 3
        {0, 1, 2, 4, 6, 3, 5, 7}, // 0b10101000 (168)|Left-PC: 5
        {1, 2, 4, 6, 0, 3, 5, 7}, // 0b10101001 (169)|Left-PC: 4
        {0, 2, 4, 6, 1, 3, 5, 7}, // 0b10101010 (170)|Left-PC: 4
        {2, 4, 6, 0, 1, 3, 5, 7}, // 0b10101011 (171)|Left-PC: 3
        {0, 1, 4, 6, 2, 3, 5, 7}, // 0b10101100 (172)|Left-PC: 4
        {1, 4, 6, 0, 2, 3, 5, 7}, // 0b10101101 (173)|Left-PC: 3
        {0, 4, 6, 1, 2, 3, 5, 7}, // 0b10101110 (174)|Left-PC: 3
        {4, 6, 0, 1, 2, 3, 5, 7}, // 0b10101111 (175)|Left-PC: 2
        {0, 1, 2, 3, 6, 4, 5, 7}, // 0b10110000 (176)|Left-PC: 5
        {1, 2, 3, 6, 0, 4, 5, 7}, // 0b10110001 (177)|Left-PC: 4
        {0, 2, 3, 6, 1, 4, 5, 7}, // 0b10110010 (178)|Left-PC: 4
        {2, 3, 6, 0, 1, 4, 5, 7}, // 0b10110011 (179)|Left-PC: 3
        {0, 1, 3, 6, 2, 4, 5, 7}, // 0b10110100 (180)|Left-PC: 4
        {1, 3, 6, 0, 2, 4, 5, 7}, // 0b10110101 (181)|Left-PC: 3
        {0, 3, 6, 1, 2, 4, 5, 7}, // 0b10110110 (182)|Left-PC: 3
        {3, 6, 0, 1, 2, 4, 5, 7}, // 0b10110111 (183)|Left-PC: 2
        {0, 1, 2, 6, 3, 4, 5, 7}, // 0b10111000 (184)|Left-PC: 4
        {1, 2, 6, 0, 3, 4, 5, 7}, // 0b10111001 (185)|Left-PC: 3
        {0, 2, 6, 1, 3, 4, 5, 7}, // 0b10111010 (186)|Left-PC: 3
        {2, 6, 0, 1, 3, 4, 5, 7}, // 0b10111011 (187)|Left-PC: 2
        {0, 1, 6, 2, 3, 4, 5, 7}, // 0b10111100 (188)|Left-PC: 3
        {1, 6, 0, 2, 3, 4, 5, 7}, // 0b10111101 (189)|Left-PC: 2
        {0, 6, 1, 2, 3, 4, 5, 7}, // 0b10111110 (190)|Left-PC: 2
        {6, 0, 1, 2, 3, 4, 5, 7}, // 0b10111111 (191)|Left-PC: 1
        {0, 1, 2, 3, 4, 5, 6, 7}, // 0b11000000 (192)|Left-PC: 6
        {1, 2, 3, 4, 5, 0, 6, 7}, // 0b11000001 (193)|Left-PC: 5
        {0, 2, 3, 4, 5, 1, 6, 7}, // 0b11000010 (194)|Left-PC: 5
        {2, 3, 4, 5, 0, 1, 6, 7}, // 0b11000011 (195)|Left-PC: 4
        {0, 1, 3, 4, 5, 2, 6, 7}, // 0b11000100 (196)|Left-PC: 5
        {1, 3, 4, 5, 0, 2, 6, 7}, // 0b11000101 (197)|Left-PC: 4
        {0, 3, 4, 5, 1, 2, 6, 7}, // 0b11000110 (198)|Left-PC: 4
        {3, 4, 5, 0, 1, 2, 6, 7}, // 0b11000111 (199)|Left-PC: 3
        {0, 1, 2, 4, 5, 3, 6, 7}, // 0b11001000 (200)|Left-PC: 5
        {1, 2, 4, 5, 0, 3, 6, 7}, // 0b11001001 (201)|Left-PC: 4
        {0, 2, 4, 5, 1, 3, 6, 7}, // 0b11001010 (202)|Left-PC: 4
        {2, 4, 5, 0, 1, 3, 6, 7}, // 0b11001011 (203)|Left-PC: 3
        {0, 1, 4, 5, 2, 3, 6, 7}, // 0b11001100 (204)|Left-PC: 4
        {1, 4, 5, 0, 2, 3, 6, 7}, // 0b11001101 (205)|Left-PC: 3
        {0, 4, 5, 1, 2, 3, 6, 7}, // 0b11001110 (206)|Left-PC: 3
        {4, 5, 0, 1, 2, 3, 6, 7}, // 0b11001111 (207)|Left-PC: 2
        {0, 1, 2, 3, 5, 4, 6, 7}, // 0b11010000 (208)|Left-PC: 5
        {1, 2, 3, 5, 0, 4, 6, 7}, // 0b11010001 (209)|Left-PC: 4
        {0, 2, 3, 5, 1, 4, 6, 7}, // 0b11010010 (210)|Left-PC: 4
        {2, 3, 5, 0, 1, 4, 6, 7}, // 0b11010011 (211)|Left-PC: 3
        {0, 1, 3, 5, 2, 4, 6, 7}, // 0b11010100 (212)|Left-PC: 4
        {1, 3, 5, 0, 2, 4, 6, 7}, // 0b11010101 (213)|Left-PC: 3
        {0, 3, 5, 1, 2, 4, 6, 7}, // 0b11010110 (214)|Left-PC: 3
        {3, 5, 0, 1, 2, 4, 6, 7}, // 0b11010111 (215)|Left-PC: 2
        {0, 1, 2, 5, 3, 4, 6, 7}, // 0b11011000 (216)|Left-PC: 4
        {1, 2, 5, 0, 3, 4, 6, 7}, // 0b11011001 (217)|Left-PC: 3
        {0, 2, 5, 1, 3, 4, 6, 7}, // 0b11011010 (218)|Left-PC: 3
        {2, 5, 0, 1, 3, 4, 6, 7}, // 0b11011011 (219)|Left-PC: 2
        {0, 1, 5, 2, 3, 4, 6, 7}, // 0b11011100 (220)|Left-PC: 3
        {1, 5, 0, 2, 3, 4, 6, 7}, // 0b11011101 (221)|Left-PC: 2
        {0, 5, 1, 2, 3, 4, 6, 7}, // 0b11011110 (222)|Left-PC: 2
        {5, 0, 1, 2, 3, 4, 6, 7}, // 0b11011111 (223)|Left-PC: 1
        {0, 1, 2, 3, 4, 5, 6, 7}, // 0b11100000 (224)|Left-PC: 5
        {1, 2, 3, 4, 0, 5, 6, 7}, // 0b11100001 (225)|Left-PC: 4
        {0, 2, 3, 4, 1, 5, 6, 7}, // 0b11100010 (226)|Left-PC: 4
        {2, 3, 4, 0, 1, 5, 6, 7}, // 0b11100011 (227)|Left-PC: 3
        {0, 1, 3, 4, 2, 5, 6, 7}, // 0b11100100 (228)|Left-PC: 4
        {1, 3, 4, 0, 2, 5, 6, 7}, // 0b11100101 (229)|Left-PC: 3
        {0, 3, 4, 1, 2, 5, 6, 7}, // 0b11100110 (230)|Left-PC: 3
        {3, 4, 0, 1, 2, 5, 6, 7}, // 0b11100111 (231)|Left-PC: 2
        {0, 1, 2, 4, 3, 5, 6, 7}, // 0b11101000 (232)|Left-PC: 4
        {1, 2, 4, 0, 3, 5, 6, 7}, // 0b11101001 (233)|Left-PC: 3
        {0, 2, 4, 1, 3, 5, 6, 7}, // 0b11101010 (234)|Left-PC: 3
        {2, 4, 0, 1, 3, 5, 6, 7}, // 0b11101011 (235)|Left-PC: 2
        {0, 1, 4, 2, 3, 5, 6, 7}, // 0b11101100 (236)|Left-PC: 3
        {1, 4, 0, 2, 3, 5, 6, 7}, // 0b11101101 (237)|Left-PC: 2
        {0, 4, 1, 2, 3, 5, 6, 7}, // 0b11101110 (238)|Left-PC: 2
        {4, 0, 1, 2, 3, 5, 6, 7}, // 0b11101111 (239)|Left-PC: 1
        {0, 1, 2, 3, 4, 5, 6, 7}, // 0b11110000 (240)|Left-PC: 4
        {1, 2, 3, 0, 4, 5, 6, 7}, // 0b11110001 (241)|Left-PC: 3
        {0, 2, 3, 1, 4, 5, 6, 7}, // 0b11110010 (242)|Left-PC: 3
        {2, 3, 0, 1, 4, 5, 6, 7}, // 0b11110011 (243)|Left-PC: 2
        {0, 1, 3, 2, 4, 5, 6, 7}, // 0b11110100 (244)|Left-PC: 3
        {1, 3, 0, 2, 4, 5, 6, 7}, // 0b11110101 (245)|Left-PC: 2
        {0, 3, 1, 2, 4, 5, 6, 7}, // 0b11110110 (246)|Left-PC: 2
        {3, 0, 1, 2, 4, 5, 6, 7}, // 0b11110111 (247)|Left-PC: 1
        {0, 1, 2, 3, 4, 5, 6, 7}, // 0b11111000 (248)|Left-PC: 3
        {1, 2, 0, 3, 4, 5, 6, 7}, // 0b11111001 (249)|Left-PC: 2
        {0, 2, 1, 3, 4, 5, 6, 7}, // 0b11111010 (250)|Left-PC: 2
        {2, 0, 1, 3, 4, 5, 6, 7}, // 0b11111011 (251)|Left-PC: 1
        {0, 1, 2, 3, 4, 5, 6, 7}, // 0b11111100 (252)|Left-PC: 2
        {1, 0, 2, 3, 4, 5, 6, 7}, // 0b11111101 (253)|Left-PC: 1
        {0, 1, 2, 3, 4, 5, 6, 7}, // 0b11111110 (254)|Left-PC: 1
        {0, 1, 2, 3, 4, 5, 6, 7}  // 0b11111111 (255)|Left-PC: 0
    };
};