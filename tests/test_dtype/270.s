
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <E_4n118>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000170C: F4040000 F8000010
	s_mov_b32 s2, s15                                          // 000000001714: BE82000F
	s_ashr_i32 s3, s15, 31                                     // 000000001718: 86039F0F
	v_mov_b32_e32 v0, 0                                        // 00000000171C: 7E000280
	s_lshl_b64 s[2:3], s[2:3], 1                               // 000000001720: 84828102
	s_waitcnt lgkmcnt(0)                                       // 000000001724: BF89FC07
	s_add_u32 s6, s6, s2                                       // 000000001728: 80060206
	s_addc_u32 s7, s7, s3                                      // 00000000172C: 82070307
	s_add_u32 s0, s0, s2                                       // 000000001730: 80000200
	global_load_u16 v1, v0, s[6:7]                             // 000000001734: DC4A0000 01060000
	s_addc_u32 s1, s1, s3                                      // 00000000173C: 82010301
	global_load_u16 v2, v0, s[0:1]                             // 000000001740: DC4A0000 02000000
	s_add_u32 s0, s4, s2                                       // 000000001748: 80000204
	s_addc_u32 s1, s5, s3                                      // 00000000174C: 82010305
	s_waitcnt vmcnt(1)                                         // 000000001750: BF8907F7
	v_cvt_f16_u16_e32 v1, v1                                   // 000000001754: 7E02A101 ; Error: VGPR_32_Lo128: unknown register 247
	s_waitcnt vmcnt(0)                                         // 000000001758: BF8903F7
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000175C: BF870001
	v_add_f16_e32 v1, v2, v1                                   // 000000001760: 64020302
	global_store_b16 v0, v1, s[0:1]                            // 000000001764: DC660000 00000100
	s_nop 0                                                    // 00000000176C: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001770: BFB60003
	s_endpgm                                                   // 000000001774: BFB00000
