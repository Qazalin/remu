
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_3520_14_2_2n1>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001700: F4080000 F8000000
	s_mul_i32 s4, s15, 56                                      // 000000001708: 9604B80F
	v_dual_mov_b32 v0, 0xff800000 :: v_dual_mov_b32 v1, 0      // 00000000170C: CA1000FF 00000080 FF800000
	s_ashr_i32 s5, s4, 31                                      // 000000001718: 86059F04
	s_mul_i32 s6, s15, 14                                      // 00000000171C: 96068E0F
	s_lshl_b64 s[4:5], s[4:5], 2                               // 000000001720: 84848204
	s_waitcnt lgkmcnt(0)                                       // 000000001724: BF89FC07
	s_add_u32 s4, s2, s4                                       // 000000001728: 80040402
	s_addc_u32 s5, s3, s5                                      // 00000000172C: 82050503
	s_lshl_b32 s2, s14, 1                                      // 000000001730: 8402810E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001734: BF870499
	s_ashr_i32 s3, s2, 31                                      // 000000001738: 86039F02
	s_lshl_b64 s[2:3], s[2:3], 2                               // 00000000173C: 84828202
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001740: BF870009
	s_add_u32 s2, s4, s2                                       // 000000001744: 80020204
	s_addc_u32 s3, s5, s3                                      // 000000001748: 82030305
	s_clause 0x1                                               // 00000000174C: BF850001
	s_load_b64 s[4:5], s[2:3], null                            // 000000001750: F4040101 F8000000
	s_load_b64 s[2:3], s[2:3], 0x70                            // 000000001758: F4040081 F8000070
	s_ashr_i32 s7, s6, 31                                      // 000000001760: 86079F06
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001764: BF870499
	s_lshl_b64 s[6:7], s[6:7], 2                               // 000000001768: 84868206
	s_add_u32 s6, s0, s6                                       // 00000000176C: 80060600
	s_waitcnt lgkmcnt(0)                                       // 000000001770: BF89FC07
	v_max3_f32 v0, s5, s4, v0                                  // 000000001774: D61C0000 04000805
	s_addc_u32 s4, s1, s7                                      // 00000000177C: 82040701
	s_ashr_i32 s15, s14, 31                                    // 000000001780: 860F9F0E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001784: BF870099
	s_lshl_b64 s[0:1], s[14:15], 2                             // 000000001788: 8480820E
	v_max3_f32 v0, s3, s2, v0                                  // 00000000178C: D61C0000 04000403
	s_add_u32 s0, s6, s0                                       // 000000001794: 80000006
	s_addc_u32 s1, s4, s1                                      // 000000001798: 82010104
	global_store_b32 v1, v0, s[0:1]                            // 00000000179C: DC6A0000 00000001
	s_nop 0                                                    // 0000000017A4: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000017A8: BFB60003
	s_endpgm                                                   // 0000000017AC: BFB00000
