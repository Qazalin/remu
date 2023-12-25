
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <E_8_24_9>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001700: F4080000 F8000000
	s_mul_hi_i32 s5, s13, 0x55555556                           // 000000001708: 9705FF0D 55555556
	s_mov_b32 s4, s13                                          // 000000001710: BE84000D
	s_lshr_b32 s6, s5, 31                                      // 000000001714: 85069F05
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001718: BF870499
	s_add_i32 s5, s5, s6                                       // 00000000171C: 81050605
	s_mul_i32 s5, s5, 3                                        // 000000001720: 96058305
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000001724: BF8704B9
	s_sub_i32 s6, s13, s5                                      // 000000001728: 8186050D
	s_mul_hi_i32 s5, s14, 0x2aaaaaab                           // 00000000172C: 9705FF0E 2AAAAAAB
	s_ashr_i32 s7, s6, 31                                      // 000000001734: 86079F06
	s_lshl_b64 s[6:7], s[6:7], 2                               // 000000001738: 84868206
	s_waitcnt lgkmcnt(0)                                       // 00000000173C: BF89FC07
	s_add_u32 s6, s2, s6                                       // 000000001740: 80060602
	s_addc_u32 s7, s3, s7                                      // 000000001744: 82070703
	s_lshr_b32 s2, s5, 31                                      // 000000001748: 85029F05
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000174C: BF870499
	s_add_i32 s2, s5, s2                                       // 000000001750: 81020205
	s_mul_i32 s2, s2, 6                                        // 000000001754: 96028602
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001758: BF870499
	s_sub_i32 s2, s14, s2                                      // 00000000175C: 8182020E
	s_mul_i32 s2, s2, 3                                        // 000000001760: 96028302
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001764: BF870499
	s_ashr_i32 s3, s2, 31                                      // 000000001768: 86039F02
	s_lshl_b64 s[2:3], s[2:3], 2                               // 00000000176C: 84828202
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000001770: BF8704B9
	s_add_u32 s5, s6, s2                                       // 000000001774: 80050206
	s_addc_u32 s6, s7, s3                                      // 000000001778: 82060307
	s_ashr_i32 s2, s15, 31                                     // 00000000177C: 86029F0F
	s_lshr_b32 s2, s2, 30                                      // 000000001780: 85029E02
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001784: BF870499
	s_add_i32 s2, s15, s2                                      // 000000001788: 8102020F
	s_and_b32 s2, s2, -4                                       // 00000000178C: 8B02C402
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001790: BF870499
	s_sub_i32 s2, s15, s2                                      // 000000001794: 8182020F
	s_mul_i32 s2, s2, 18                                       // 000000001798: 96029202
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000179C: BF870499
	s_ashr_i32 s3, s2, 31                                      // 0000000017A0: 86039F02
	s_lshl_b64 s[2:3], s[2:3], 2                               // 0000000017A4: 84828202
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000017A8: BF870009
	s_add_u32 s2, s5, s2                                       // 0000000017AC: 80020205
	s_addc_u32 s3, s6, s3                                      // 0000000017B0: 82030306
	s_load_b32 s6, s[2:3], null                                // 0000000017B4: F4000181 F8000000
	s_mul_i32 s2, s15, 0xd8                                    // 0000000017BC: 9602FF0F 000000D8
	v_mov_b32_e32 v0, 0                                        // 0000000017C4: 7E000280
	s_ashr_i32 s3, s2, 31                                      // 0000000017C8: 86039F02
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000017CC: BF870499
	s_lshl_b64 s[2:3], s[2:3], 2                               // 0000000017D0: 84828202
	s_add_u32 s2, s0, s2                                       // 0000000017D4: 80020200
	s_mul_i32 s0, s14, 9                                       // 0000000017D8: 9600890E
	s_addc_u32 s3, s1, s3                                      // 0000000017DC: 82030301
	s_ashr_i32 s1, s0, 31                                      // 0000000017E0: 86019F00
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000017E4: BF870499
	s_lshl_b64 s[0:1], s[0:1], 2                               // 0000000017E8: 84808200
	s_add_u32 s2, s2, s0                                       // 0000000017EC: 80020002
	s_addc_u32 s3, s3, s1                                      // 0000000017F0: 82030103
	s_ashr_i32 s5, s13, 31                                     // 0000000017F4: 86059F0D
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000017F8: BF870009
	s_lshl_b64 s[0:1], s[4:5], 2                               // 0000000017FC: 84808204
	s_waitcnt lgkmcnt(0)                                       // 000000001800: BF89FC07
	v_mov_b32_e32 v1, s6                                       // 000000001804: 7E020206
	s_add_u32 s0, s2, s0                                       // 000000001808: 80000002
	s_addc_u32 s1, s3, s1                                      // 00000000180C: 82010103
	global_store_b32 v0, v1, s[0:1]                            // 000000001810: DC6A0000 00000100
	s_nop 0                                                    // 000000001818: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 00000000181C: BFB60003
	s_endpgm                                                   // 000000001820: BFB00000
