
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_2368_9_3_3>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001700: F4080000 F8000000
	s_mul_i32 s4, s15, 0x54                                    // 000000001708: 9604FF0F 00000054
	s_mul_i32 s6, s14, 3                                       // 000000001710: 9606830E
	s_ashr_i32 s5, s4, 31                                      // 000000001714: 86059F04
	v_mov_b32_e32 v1, 0                                        // 000000001718: 7E020280
	s_lshl_b64 s[4:5], s[4:5], 2                               // 00000000171C: 84848204
	s_waitcnt lgkmcnt(0)                                       // 000000001720: BF89FC07
	s_add_u32 s4, s2, s4                                       // 000000001724: 80040402
	s_addc_u32 s5, s3, s5                                      // 000000001728: 82050503
	s_ashr_i32 s7, s6, 31                                      // 00000000172C: 86079F06
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001730: BF870499
	s_lshl_b64 s[2:3], s[6:7], 2                               // 000000001734: 84828206
	s_add_u32 s2, s4, s2                                       // 000000001738: 80020204
	s_addc_u32 s3, s5, s3                                      // 00000000173C: 82030305
	s_clause 0x3                                               // 000000001740: BF850003
	s_load_b64 s[4:5], s[2:3], null                            // 000000001744: F4040101 F8000000
	s_load_b32 s8, s[2:3], 0x8                                 // 00000000174C: F4000201 F8000008
	s_load_b64 s[6:7], s[2:3], 0x70                            // 000000001754: F4040181 F8000070
	s_load_b32 s9, s[2:3], 0x78                                // 00000000175C: F4000241 F8000078
	s_waitcnt lgkmcnt(0)                                       // 000000001764: BF89FC07
	v_add_f32_e64 v0, s4, 0                                    // 000000001768: D5030000 00010004
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001770: BF8700A1
	v_add_f32_e32 v0, s5, v0                                   // 000000001774: 06000005
	s_load_b64 s[4:5], s[2:3], 0xe0                            // 000000001778: F4040101 F80000E0
	v_add_f32_e32 v0, s8, v0                                   // 000000001780: 06000008
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000001784: BF8704B1
	v_add_f32_e32 v0, s6, v0                                   // 000000001788: 06000006
	s_load_b32 s6, s[2:3], 0xe8                                // 00000000178C: F4000181 F80000E8
	s_mul_i32 s2, s15, 9                                       // 000000001794: 9602890F
	s_ashr_i32 s3, s2, 31                                      // 000000001798: 86039F02
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 00000000179C: BF8704A1
	v_add_f32_e32 v0, s7, v0                                   // 0000000017A0: 06000007
	s_lshl_b64 s[2:3], s[2:3], 2                               // 0000000017A4: 84828202
	s_add_u32 s2, s0, s2                                       // 0000000017A8: 80020200
	s_addc_u32 s3, s1, s3                                      // 0000000017AC: 82030301
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 0000000017B0: BF8704A1
	v_add_f32_e32 v0, s9, v0                                   // 0000000017B4: 06000009
	s_ashr_i32 s15, s14, 31                                    // 0000000017B8: 860F9F0E
	s_lshl_b64 s[0:1], s[14:15], 2                             // 0000000017BC: 8480820E
	s_waitcnt lgkmcnt(0)                                       // 0000000017C0: BF89FC07
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 0000000017C4: BF8700B1
	v_add_f32_e32 v0, s4, v0                                   // 0000000017C8: 06000004
	s_add_u32 s0, s2, s0                                       // 0000000017CC: 80000002
	s_addc_u32 s1, s3, s1                                      // 0000000017D0: 82010103
	v_add_f32_e32 v0, s5, v0                                   // 0000000017D4: 06000005
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017D8: BF870091
	v_add_f32_e32 v0, s6, v0                                   // 0000000017DC: 06000006
	v_mul_f32_e32 v0, 0x3de38e39, v0                           // 0000000017E0: 100000FF 3DE38E39
	global_store_b32 v1, v0, s[0:1]                            // 0000000017E8: DC6A0000 00000001
	s_nop 0                                                    // 0000000017F0: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000017F4: BFB60003
	s_endpgm                                                   // 0000000017F8: BFB00000
